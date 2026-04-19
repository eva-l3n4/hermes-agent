"""Tests for acp_adapter.events — callback factories for ACP notifications."""

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import acp
from acp.schema import ToolCallStart, ToolCallProgress, AgentThoughtChunk, AgentMessageChunk

from acp_adapter.events import (
    make_message_cb,
    make_step_cb,
    make_thinking_cb,
    make_tool_progress_cb,
)


@pytest.fixture()
def mock_conn():
    """Mock ACP Client connection."""
    conn = MagicMock(spec=acp.Client)
    conn.session_update = AsyncMock()
    return conn


@pytest.fixture()
def event_loop_fixture():
    """Create a real event loop for testing threadsafe coroutine submission."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tool progress callback
# ---------------------------------------------------------------------------


class TestToolProgressCallback:
    def test_emits_tool_call_start(self, mock_conn, event_loop_fixture):
        """Tool progress should emit a ToolCallStart update."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        # Run callback in the event loop context
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ ls -la", {"command": "ls -la"})

        # Should have tracked the tool call ID
        assert "terminal" in tool_call_ids

        # Should have called run_coroutine_threadsafe
        mock_rcts.assert_called_once()
        coro = mock_rcts.call_args[0][0]
        # The coroutine should be conn.session_update
        assert mock_conn.session_update.called or coro is not None

    def test_handles_string_args(self, mock_conn, event_loop_fixture):
        """If args is a JSON string, it should be parsed."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "read_file", "Reading /etc/hosts", '{"path": "/etc/hosts"}')

        assert "read_file" in tool_call_ids

    def test_handles_non_dict_args(self, mock_conn, event_loop_fixture):
        """If args is not a dict, it should be wrapped."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("tool.started", "terminal", "$ echo hi", None)

        assert "terminal" in tool_call_ids

    def test_duplicate_same_name_tool_calls_use_fifo_ids(self, mock_conn, event_loop_fixture):
        """Multiple same-name tool calls should be tracked independently in order."""
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        progress_cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
        step_cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            progress_cb("tool.started", "terminal", "$ ls", {"command": "ls"})
            progress_cb("tool.started", "terminal", "$ pwd", {"command": "pwd"})
            assert len(tool_call_ids["terminal"]) == 2

            step_cb(1, [{"name": "terminal", "result": "ok-1"}])
            assert len(tool_call_ids["terminal"]) == 1

            step_cb(2, [{"name": "terminal", "result": "ok-2"}])
            assert "terminal" not in tool_call_ids


# ---------------------------------------------------------------------------
# Thinking callback
# ---------------------------------------------------------------------------


class TestThinkingCallback:
    def test_emits_thought_chunk(self, mock_conn, event_loop_fixture):
        """Thinking callback should emit AgentThoughtChunk."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Analyzing the code...")

        mock_rcts.assert_called_once()

    def test_ignores_empty_text(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_thinking_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Step callback
# ---------------------------------------------------------------------------


class TestStepCallback:
    def test_completes_tracked_tool_calls(self, mock_conn, event_loop_fixture):
        """Step callback should mark tracked tools as completed."""
        tool_call_ids = {"terminal": "tc-abc123"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "terminal", "result": "success"}])

        # Tool should have been removed from tracking
        assert "terminal" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_ignores_untracked_tools(self, mock_conn, event_loop_fixture):
        """Tools not in tool_call_ids should be silently ignored."""
        tool_call_ids = {}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb(1, [{"name": "unknown_tool", "result": "ok"}])

        mock_rcts.assert_not_called()

    def test_handles_string_tool_info(self, mock_conn, event_loop_fixture):
        """Tool info as a string (just the name) should work."""
        tool_call_ids = {"read_file": "tc-def456"}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(2, ["read_file"])

        assert "read_file" not in tool_call_ids
        mock_rcts.assert_called_once()

    def test_result_passed_to_build_tool_complete(self, mock_conn, event_loop_fixture):
        """Tool result from prev_tools dict is forwarded to build_tool_complete."""
        from collections import deque

        tool_call_ids = {"terminal": deque(["tc-xyz789"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            # Provide a result string in the tool info dict
            cb(1, [{"name": "terminal", "result": '{"output": "hello"}'}])

        mock_btc.assert_called_once_with(
            "tc-xyz789", "terminal", result='{"output": "hello"}', function_args=None, snapshot=None
        )

    def test_none_result_passed_through(self, mock_conn, event_loop_fixture):
        """When result is None (e.g. first iteration), None is passed through."""
        from collections import deque

        tool_call_ids = {"web_search": deque(["tc-aaa"])}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, {})

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "web_search", "result": None}])

        mock_btc.assert_called_once_with("tc-aaa", "web_search", result=None, function_args=None, snapshot=None)

    def test_step_callback_passes_arguments_and_snapshot(self, mock_conn, event_loop_fixture):
        from collections import deque

        tool_call_ids = {"write_file": deque(["tc-write"])}
        tool_call_meta = {"tc-write": {"args": {"path": "fallback.txt"}, "snapshot": "snap"}}
        loop = event_loop_fixture

        cb = make_step_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts, \
             patch("acp_adapter.events.build_tool_complete") as mock_btc:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb(1, [{"name": "write_file", "result": '{"bytes_written": 23}', "arguments": {"path": "diff-test.txt"}}])

        mock_btc.assert_called_once_with(
            "tc-write",
            "write_file",
            result='{"bytes_written": 23}',
            function_args={"path": "diff-test.txt"},
            snapshot="snap",
        )

    def test_tool_progress_captures_snapshot_metadata(self, mock_conn, event_loop_fixture):
        tool_call_ids = {}
        tool_call_meta = {}
        loop = event_loop_fixture

        with patch("acp_adapter.events.make_tool_call_id", return_value="tc-meta"), \
             patch("acp_adapter.events._send_update") as mock_send, \
             patch("agent.display.capture_local_edit_snapshot", return_value="snapshot"):
            cb = make_tool_progress_cb(mock_conn, "session-1", loop, tool_call_ids, tool_call_meta)
            cb("tool.started", "write_file", None, {"path": "diff-test.txt", "content": "hello"})

        assert list(tool_call_ids["write_file"]) == ["tc-meta"]
        assert tool_call_meta["tc-meta"] == {
            "args": {"path": "diff-test.txt", "content": "hello"},
            "snapshot": "snapshot",
        }
        mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# Message callback
# ---------------------------------------------------------------------------


class TestMessageCallback:
    def test_emits_agent_message_chunk(self, mock_conn, event_loop_fixture):
        """Message callback should emit AgentMessageChunk."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future

            cb("Here is your answer.")

        mock_rcts.assert_called_once()

    def test_ignores_empty_message(self, mock_conn, event_loop_fixture):
        """Empty text should not emit any update."""
        loop = event_loop_fixture

        cb = make_message_cb(mock_conn, "session-1", loop)

        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            cb("")

        mock_rcts.assert_not_called()


# ---------------------------------------------------------------------------
# Subagent bridge tests
# ---------------------------------------------------------------------------


class TestSubagentBridge:
    """Verify that subagent.* events are bridged to _hermes/subagent_update
    notifications, and that subagent.progress / tool.started are not."""

    def _make_cb(self, monkeypatch):
        """Create a tool_progress_cb with _send_notification stubbed."""
        calls = []

        def fake_send_notification(conn, loop, method, params):
            calls.append((method, params))

        monkeypatch.setattr(
            "acp_adapter.events._send_notification",
            fake_send_notification,
        )
        cb = make_tool_progress_cb(
            conn=MagicMock(spec=acp.Client),
            session_id="parent-sid",
            loop=None,
            tool_call_ids={},
            tool_call_meta={},
        )
        return cb, calls

    def test_subagent_start_emits_update(self, monkeypatch):
        cb, calls = self._make_cb(monkeypatch)
        cb(
            "subagent.start",
            name="delegate_tool",
            preview=None,
            args={},
            goal="do the thing",
            child_session_id="child-sid",
            task_index=0,
            task_count=1,
        )
        assert len(calls) == 1
        method, params = calls[0]
        assert method == "hermes/subagent_update"
        assert params["session_id"] == "parent-sid"
        assert params["child_session_id"] == "child-sid"
        assert params["event_type"] == "start"
        assert params["goal"] == "do the thing"
        assert params["task_index"] == 0
        assert params["task_count"] == 1

    def test_subagent_thinking_emits_update(self, monkeypatch):
        cb, calls = self._make_cb(monkeypatch)
        cb(
            "subagent.thinking",
            name="delegate_tool",
            preview="reasoning about the problem",
            args={},
            child_session_id="child-sid",
            task_index=0,
            task_count=1,
        )
        assert len(calls) == 1
        method, params = calls[0]
        assert method == "hermes/subagent_update"
        assert params["event_type"] == "thinking"
        assert params["preview"] == "reasoning about the problem"
        assert params["child_session_id"] == "child-sid"

    def test_subagent_tool_emits_update(self, monkeypatch):
        cb, calls = self._make_cb(monkeypatch)
        cb(
            "subagent.tool",
            name="terminal",
            preview="$ ls -la",
            args={"command": "ls", "path": "/tmp"},
            child_session_id="child-sid",
            task_index=0,
            task_count=1,
        )
        assert len(calls) == 1
        method, params = calls[0]
        assert method == "hermes/subagent_update"
        assert params["event_type"] == "tool"
        assert params["tool_name"] == "terminal"
        assert params["preview"] == "$ ls -la"
        assert params["args"] == {"command": "ls", "path": "/tmp"}

    def test_subagent_complete_emits_update(self, monkeypatch):
        cb, calls = self._make_cb(monkeypatch)
        cb(
            "subagent.complete",
            name="delegate_tool",
            preview="all done",
            args={},
            status="success",
            child_session_id="child-sid",
            task_index=0,
            task_count=1,
            duration_seconds=4.2,
        )
        assert len(calls) == 1
        method, params = calls[0]
        assert method == "hermes/subagent_update"
        assert params["event_type"] == "complete"
        assert params["status"] == "success"
        assert params["summary"] == "all done"
        assert params["duration_seconds"] == 4.2

    def test_subagent_progress_not_bridged(self, monkeypatch):
        """subagent.progress is CLI-only formatting and must NOT be bridged."""
        cb, calls = self._make_cb(monkeypatch)
        cb(
            "subagent.progress",
            name="delegate_tool",
            preview="some progress",
            args={},
            child_session_id="child-sid",
            task_index=0,
            task_count=1,
        )
        assert len(calls) == 0

    def test_tool_started_not_bridged(self, monkeypatch):
        """Regression: tool.started must NOT be bridged as subagent_update."""
        cb, calls = self._make_cb(monkeypatch)
        with patch("acp_adapter.events.asyncio.run_coroutine_threadsafe") as mock_rcts:
            future = MagicMock(spec=Future)
            future.result.return_value = None
            mock_rcts.return_value = future
            cb(
                "tool.started",
                name="terminal",
                preview="$ ls",
                args={"command": "ls"},
            )
        # No subagent notification should have been emitted
        assert len(calls) == 0
        # But the original path should still work (run_coroutine_threadsafe called)
        mock_rcts.assert_called_once()
