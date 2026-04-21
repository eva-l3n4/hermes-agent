"""Callback factories for bridging AIAgent events to ACP notifications.

Each factory returns a callable with the signature that AIAgent expects
for its callbacks. Internally, the callbacks push ACP session updates
to the client via ``conn.session_update()`` using
``asyncio.run_coroutine_threadsafe()`` (since AIAgent runs in a worker
thread while the event loop lives on the main thread).
"""

import asyncio
import json
import logging
from collections import deque
from typing import Any, Callable, Deque, Dict

import acp

from .tools import (
    build_tool_complete,
    build_tool_start,
    make_tool_call_id,
)

logger = logging.getLogger(__name__)


def _send_update(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    update: Any,
) -> None:
    """Fire-and-forget an ACP session update from a worker thread."""
    try:
        future = asyncio.run_coroutine_threadsafe(
            conn.session_update(session_id, update), loop
        )
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send ACP update", exc_info=True)


def _send_notification(
    conn: acp.Client,
    loop: asyncio.AbstractEventLoop,
    method: str,
    params: dict,
) -> None:
    """Fire-and-forget a JSON-RPC notification from a worker thread.

    ``method`` should be given without the leading underscore; the ACP
    ``AgentSideConnection.ext_notification`` helper prepends it automatically
    (so ``"hermes/subagent_update"`` goes on the wire as
    ``"_hermes/subagent_update"``, matching the extension namespace convention).
    """
    try:
        future = asyncio.run_coroutine_threadsafe(
            conn.ext_notification(method, params), loop
        )
        future.result(timeout=5)
    except Exception:
        logger.debug("Failed to send notification %s", method, exc_info=True)


# NOTE: This bridge assumes at most ONE level of delegation
# (parent -> child, no grandchildren), matching delegate_tool.MAX_DEPTH = 2.
# If MAX_DEPTH is ever raised, the Kaishi-side zoom view needs updating to
# handle nested children — today it degrades gracefully (unknown
# child_session_id creates a new task line rather than being dropped).
def _emit_subagent_update(
    conn: acp.Client,
    loop: asyncio.AbstractEventLoop,
    parent_session_id: str,
    event_type: str,
    tool_name,
    preview,
    args,
    kwargs: dict,
) -> None:
    """Translate a subagent.* callback event into a _hermes/subagent_update notification."""
    short_type = event_type.split(".", 1)[1]  # "subagent.start" -> "start"
    params: dict = {
        "session_id": parent_session_id,
        "child_session_id": kwargs.get("child_session_id"),
        "task_index": kwargs.get("task_index", 0),
        "task_count": kwargs.get("task_count", 1),
        "event_type": short_type,
    }
    if short_type == "start":
        params["goal"] = kwargs.get("goal") or preview or ""
    elif short_type == "thinking":
        params["preview"] = preview or ""
    elif short_type == "tool":
        params["tool_name"] = tool_name or ""
        if preview:
            params["preview"] = preview
        if isinstance(args, dict):
            # Cap args serialization at ~2KB to avoid bloating notifications
            import json as _json
            try:
                serialized = _json.dumps(args, ensure_ascii=False)
                if len(serialized) <= 2048:
                    params["args"] = args
            except Exception:
                pass
    elif short_type == "complete":
        params["status"] = kwargs.get("status", "success")
        if preview:
            params["summary"] = preview
        if "duration_seconds" in kwargs:
            params["duration_seconds"] = kwargs["duration_seconds"]

    _send_notification(conn, loop, "hermes/subagent_update", params)


# ------------------------------------------------------------------
# Tool progress callback
# ------------------------------------------------------------------

def make_tool_progress_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
) -> Callable:
    """Create a ``tool_progress_callback`` for AIAgent.

    Signature expected by AIAgent::

        tool_progress_callback(event_type: str, name: str, preview: str, args: dict, **kwargs)

    Emits ``ToolCallStart`` for ``tool.started`` events and tracks IDs in a FIFO
    queue per tool name so duplicate/parallel same-name calls still complete
    against the correct ACP tool call.  Other event types (``tool.completed``,
    ``reasoning.available``) are silently ignored.
    """

    def _tool_progress(event_type: str, name: str = None, preview: str = None, args: Any = None, **kwargs) -> None:
        # --- Subagent bridge ---
        # Bridge subagent.* events (except subagent.progress which is CLI-only
        # formatting) to the client as _hermes/subagent_update notifications.
        if event_type.startswith("subagent.") and event_type != "subagent.progress":
            _emit_subagent_update(conn, loop, session_id, event_type, name, preview, args, kwargs)
            return

        # Only emit ACP ToolCallStart for tool.started; ignore other event types
        if event_type != "tool.started":
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}

        tc_id = make_tool_call_id()
        queue = tool_call_ids.get(name)
        if queue is None:
            queue = deque()
            tool_call_ids[name] = queue
        elif isinstance(queue, str):
            queue = deque([queue])
            tool_call_ids[name] = queue
        queue.append(tc_id)

        update = build_tool_start(tc_id, name, args)
        _send_update(conn, session_id, loop, update)

    return _tool_progress


# ------------------------------------------------------------------
# Thinking callback
# ------------------------------------------------------------------

def make_thinking_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a ``thinking_callback`` for AIAgent."""

    def _thinking(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_thought_text(text)
        _send_update(conn, session_id, loop, update)

    return _thinking


# ------------------------------------------------------------------
# Step callback
# ------------------------------------------------------------------

def make_step_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
    tool_call_ids: Dict[str, Deque[str]],
) -> Callable:
    """Create a ``step_callback`` for AIAgent.

    Signature expected by AIAgent::

        step_callback(api_call_count: int, prev_tools: list)
    """

    def _step(api_call_count: int, prev_tools: Any = None) -> None:
        if prev_tools and isinstance(prev_tools, list):
            for tool_info in prev_tools:
                tool_name = None
                result = None

                if isinstance(tool_info, dict):
                    tool_name = tool_info.get("name") or tool_info.get("function_name")
                    result = tool_info.get("result") or tool_info.get("output")
                elif isinstance(tool_info, str):
                    tool_name = tool_info

                queue = tool_call_ids.get(tool_name or "")
                if isinstance(queue, str):
                    queue = deque([queue])
                    tool_call_ids[tool_name] = queue
                if tool_name and queue:
                    tc_id = queue.popleft()
                    update = build_tool_complete(
                        tc_id, tool_name, result=str(result) if result is not None else None
                    )
                    _send_update(conn, session_id, loop, update)
                    if not queue:
                        tool_call_ids.pop(tool_name, None)

    return _step


# ------------------------------------------------------------------
# Agent message callback
# ------------------------------------------------------------------

def make_message_cb(
    conn: acp.Client,
    session_id: str,
    loop: asyncio.AbstractEventLoop,
) -> Callable:
    """Create a callback that streams agent response text to the editor."""

    def _message(text: str) -> None:
        if not text:
            return
        update = acp.update_agent_message_text(text)
        _send_update(conn, session_id, loop, update)

    return _message
