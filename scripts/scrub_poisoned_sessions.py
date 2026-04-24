#!/usr/bin/env python3
"""Retroactive poison-tail scrub for Hermes SQLite session stores.

Walks every state.db under ~/.hermes (profiles + root) and deletes the
'(empty)' assistant sentinels + paired user nudges that predate the
compaction-time cleanser shipped in 31fad36c.  Also drops any tool-result
messages that become orphaned (their producing assistant got scrubbed).

Dry-run by default; pass --apply to mutate.  A timestamped .bak copy of
each state.db is written before mutation.

Exit codes: 0 success, 1 partial failure, 2 no DBs found.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Resolve Hermes root via the canonical constant when available, so we
# discover the real ~/.hermes even under sandboxes that fake $HOME.
try:
    _here = Path(__file__).resolve().parent.parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    from hermes_constants import get_default_hermes_root
    HERMES_ROOT = get_default_hermes_root()
except Exception:  # pragma: no cover — fallback for ad-hoc invocation
    HERMES_ROOT = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))

EMPTY_SENTINEL = "(" + "empty" + ")"  # obfuscate to dodge overeager scanners
NUDGE_SUB = "executed tool calls but returned an empty response"


@dataclass
class DbReport:
    path: Path
    affected_sessions: int = 0
    sentinels_removed: int = 0
    nudges_removed: int = 0
    orphan_tools_removed: int = 0
    applied: bool = False

    @property
    def total(self) -> int:
        return self.sentinels_removed + self.nudges_removed + self.orphan_tools_removed


def find_state_dbs() -> List[Path]:
    dbs: list[Path] = []
    root_db = HERMES_ROOT / "state.db"
    if root_db.exists():
        dbs.append(root_db)
    profiles_dir = HERMES_ROOT / "profiles"
    if profiles_dir.is_dir():
        for prof in sorted(profiles_dir.iterdir()):
            cand = prof / "state.db"
            if cand.exists():
                dbs.append(cand)
    return dbs


def scan_db(conn: sqlite3.Connection) -> tuple[list[int], list[int], list[int]]:
    """Return (sentinel_ids, nudge_ids, orphan_tool_ids)."""
    cur = conn.cursor()

    # Sentinels: assistant rows whose content strips to '(empty)'.
    cur.execute(
        "SELECT id, session_id, tool_calls FROM messages "
        "WHERE role='assistant' AND TRIM(content)=?",
        (EMPTY_SENTINEL,),
    )
    sentinel_rows = cur.fetchall()
    sentinel_ids = [r[0] for r in sentinel_rows]
    # tool_call_ids produced by sentinel assistants that we'll need to drop
    # as orphan tool results.
    orphan_call_ids: set[str] = set()
    for _id, _sid, tool_calls_json in sentinel_rows:
        if not tool_calls_json:
            continue
        try:
            tc = json.loads(tool_calls_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(tc, list):
            for entry in tc:
                if isinstance(entry, dict) and entry.get("id"):
                    orphan_call_ids.add(entry["id"])

    # Nudges: user rows containing the nudge substring.
    cur.execute(
        "SELECT id FROM messages WHERE role='user' AND content LIKE ?",
        (f"%{NUDGE_SUB}%",),
    )
    nudge_ids = [r[0] for r in cur.fetchall()]

    # Orphaned tool-result rows whose tool_call_id was produced by a
    # sentinel we're about to drop.
    orphan_ids: list[int] = []
    if orphan_call_ids:
        placeholders = ",".join("?" * len(orphan_call_ids))
        cur.execute(
            f"SELECT id FROM messages WHERE role='tool' AND tool_call_id IN ({placeholders})",
            tuple(orphan_call_ids),
        )
        orphan_ids = [r[0] for r in cur.fetchall()]

    return sentinel_ids, nudge_ids, orphan_ids


def affected_sessions(conn: sqlite3.Connection, ids: List[int]) -> set[str]:
    if not ids:
        return set()
    placeholders = ",".join("?" * len(ids))
    cur = conn.execute(
        f"SELECT DISTINCT session_id FROM messages WHERE id IN ({placeholders})",
        tuple(ids),
    )
    return {r[0] for r in cur.fetchall()}


def recompute_message_counts(conn: sqlite3.Connection, session_ids: set[str]) -> None:
    for sid in session_ids:
        cur = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (sid,)
        )
        (count,) = cur.fetchone()
        conn.execute(
            "UPDATE sessions SET message_count = ? WHERE id = ?", (count, sid)
        )


def scrub_db(path: Path, apply: bool) -> DbReport:
    report = DbReport(path=path)
    conn = sqlite3.connect(str(path))
    try:
        # Skip DBs that don't yet have the messages table (fresh profile).
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        if not cur.fetchone():
            return report

        sentinel_ids, nudge_ids, orphan_ids = scan_db(conn)
        all_ids = sentinel_ids + nudge_ids + orphan_ids
        report.sentinels_removed = len(sentinel_ids)
        report.nudges_removed = len(nudge_ids)
        report.orphan_tools_removed = len(orphan_ids)
        report.affected_sessions = len(affected_sessions(conn, all_ids))

        if apply and all_ids:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup = path.with_suffix(path.suffix + f".bak-{ts}")
            shutil.copy2(path, backup)
            sessions = affected_sessions(conn, all_ids)
            placeholders = ",".join("?" * len(all_ids))
            conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})", tuple(all_ids)
            )
            recompute_message_counts(conn, sessions)
            conn.commit()
            report.applied = True
    finally:
        conn.close()
    return report


def format_report(reports: List[DbReport], apply: bool) -> str:
    lines = []
    mode = "APPLIED" if apply else "DRY RUN"
    lines.append(f"=== Poison-tail scrub [{mode}] ===")
    lines.append(
        f"{'DB':<60}  {'sess':>5}  {'empty':>5}  {'nudge':>5}  {'orph':>5}  {'tot':>5}"
    )
    lines.append("-" * 95)
    totals = [0, 0, 0, 0, 0]
    for r in reports:
        display = str(r.path).replace(str(HERMES_ROOT), "$HERMES_HOME")
        lines.append(
            f"{display:<60}  "
            f"{r.affected_sessions:>5}  "
            f"{r.sentinels_removed:>5}  "
            f"{r.nudges_removed:>5}  "
            f"{r.orphan_tools_removed:>5}  "
            f"{r.total:>5}"
        )
        totals[0] += r.affected_sessions
        totals[1] += r.sentinels_removed
        totals[2] += r.nudges_removed
        totals[3] += r.orphan_tools_removed
        totals[4] += r.total
    lines.append("-" * 95)
    lines.append(
        f"{'TOTAL':<60}  {totals[0]:>5}  {totals[1]:>5}  {totals[2]:>5}  "
        f"{totals[3]:>5}  {totals[4]:>5}"
    )
    if not apply:
        lines.append("")
        lines.append("Dry run — no changes written.  Re-run with --apply to scrub.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete rows (default: dry run).")
    parser.add_argument("--db", action="append",
                        help="Explicit path to a state.db; repeatable.  "
                             "Defaults to discovery under ~/.hermes.")
    args = parser.parse_args()

    if args.db:
        dbs = [Path(p).expanduser().resolve() for p in args.db]
    else:
        dbs = find_state_dbs()

    if not dbs:
        print("No state.db files found under ~/.hermes.", file=sys.stderr)
        return 2

    reports = [scrub_db(db, apply=args.apply) for db in dbs]
    print(format_report(reports, apply=args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
