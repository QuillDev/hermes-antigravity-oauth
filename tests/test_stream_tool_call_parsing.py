"""Behavior tests for streaming tool-call extraction and prompt budget fitting."""

from __future__ import annotations

import json
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from acp import AntigravityACPStream
from prompt import _fit_transcript_to_budget
from stream import AntigravityStream

_TC_OPEN = "<" + "tool_call>"
_TC_CLOSE = "</" + "tool_call>"


class _FakeProc:
    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = iter(lines or [])
        self.stdout = self
        self.stderr = None

    def readline(self) -> str:
        return next(self._lines, "")

    def poll(self) -> int | None:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        return 0


def _collect_acp_stream(deltas: list[str], *, has_tools: bool = True) -> tuple[str, list[tuple[str, str, str]], str | None]:
    incoming: queue.Queue = queue.Queue()
    for delta in deltas:
        incoming.put(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": delta},
                    }
                },
            }
        )
    incoming.put(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}},
        }
    )

    worker = SimpleNamespace(
        proc=_FakeProc(),
        session_id="sess-test",
        incoming=incoming,
        get_stderr=lambda: "",
        cancel_turn=lambda: None,
        close=lambda: None,
    )
    client = SimpleNamespace(_lock=threading.RLock(), _active_processes=set())
    tools = [{"type": "function", "function": {"name": "write_file"}}] if has_tools else None
    stream = AntigravityACPStream(
        worker=cast(Any, worker),
        req_id=1,
        client=client,
        model="gemini-3-flash",
        timeout=10.0,
        tools=tools,
        is_worker=False,
    )
    return _drain(stream)


def _collect_json_stream(deltas: list[str], *, has_tools: bool = True) -> tuple[str, list[tuple[str, str, str]], str | None]:
    lines = [
        json.dumps({"event": "step_update", "step_update": {"step_type": "text", "text_delta": delta}}) + "\n"
        for delta in deltas
    ]
    lines.append(json.dumps({"event": "result", "result": {"status": "SUCCESS"}}) + "\n")
    client = SimpleNamespace(
        _isolated_gemini_dir=None,
        _lock=threading.RLock(),
        _active_processes=set(),
        _terminate_process=lambda _proc: None,
    )
    tools = [{"type": "function", "function": {"name": "write_file"}}] if has_tools else None
    stream = AntigravityStream(
        proc=cast(Any, _FakeProc(lines)),
        client=client,
        model="gemini-3-flash",
        timeout=10.0,
        tools=tools,
        is_worker=False,
    )
    return _drain(stream)


def _drain(stream) -> tuple[str, list[tuple[str, str, str]], str | None]:
    content_parts: list[str] = []
    tool_calls: list[tuple[str, str, str]] = []
    finish_reason: str | None = None
    for chunk in stream:
        for choice in getattr(chunk, "choices", ()):
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                content_parts.append(delta.content)
            for tc in delta.tool_calls or ():
                tool_calls.append((tc.id, tc.function.name, tc.function.arguments))
    return "".join(content_parts), tool_calls, finish_reason


class StreamToolCallParsingTests(unittest.TestCase):
    def test_split_boundaries_yield_identical_tool_calls_in_both_transports(self) -> None:
        payload = json.dumps(
            {
                "id": "call_split_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "out.txt", "content": "hello"})},
            }
        )
        full_text = f"Working on it. {_TC_OPEN}{payload}{_TC_CLOSE} trailing chatter"

        for chunk_size in (1, 3, 7, 13, len(full_text)):
            deltas = [full_text[i : i + chunk_size] for i in range(0, len(full_text), chunk_size)]
            for runner in (_collect_acp_stream, _collect_json_stream):
                with self.subTest(runner=runner.__name__, chunk_size=chunk_size):
                    content, calls, finish_reason = runner(deltas)
                    self.assertEqual(content, "Working on it. ")
                    self.assertEqual(
                        calls,
                        [("call_split_1", "write_file", json.dumps({"path": "out.txt", "content": "hello"}))],
                    )
                    self.assertEqual(finish_reason, "tool_calls")

    def test_multiple_sequential_tool_calls_stream_cleanly(self) -> None:
        first = json.dumps(
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "a.txt"})},
            }
        )
        second = json.dumps(
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "write_file", "arguments": json.dumps({"path": "b.txt"})},
            }
        )
        full_text = f"Start {_TC_OPEN}{first}{_TC_CLOSE}\n{_TC_OPEN}{second}{_TC_CLOSE}"
        deltas = [full_text[i : i + 5] for i in range(0, len(full_text), 5)]

        for runner in (_collect_acp_stream, _collect_json_stream):
            with self.subTest(runner=runner.__name__):
                content, calls, finish_reason = runner(deltas)
                self.assertEqual(content, "Start ")
                self.assertEqual([c[0] for c in calls], ["call_1", "call_2"])
                self.assertEqual(finish_reason, "tool_calls")

    def test_unclosed_tool_call_tag_flushes_as_plain_content(self) -> None:
        text = f"Here is a literal tag: {_TC_OPEN}not valid json"
        deltas = [text[i : i + 4] for i in range(0, len(text), 4)]
        for runner in (_collect_acp_stream, _collect_json_stream):
            with self.subTest(runner=runner.__name__):
                content, calls, finish_reason = runner(deltas)
                self.assertEqual(content, text)
                self.assertEqual(calls, [])
                self.assertEqual(finish_reason, "stop")

    def test_large_tool_call_payload_parses_intact(self) -> None:
        big_body = "line = 12345\n" * 4000
        args_json = json.dumps({"path": "big.py", "content": big_body})
        payload = json.dumps(
            {
                "id": "call_big",
                "type": "function",
                "function": {"name": "write_file", "arguments": args_json},
            }
        )
        full_text = f"Writing.\n{_TC_OPEN}{payload}{_TC_CLOSE}"
        deltas = [full_text[i : i + 16] for i in range(0, len(full_text), 16)]

        for runner in (_collect_acp_stream, _collect_json_stream):
            with self.subTest(runner=runner.__name__):
                content, calls, finish_reason = runner(deltas)
                self.assertEqual(content, "Writing.\n")
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0], ("call_big", "write_file", args_json))
                self.assertEqual(finish_reason, "tool_calls")


class PromptBudgetFittingTests(unittest.TestCase):
    def test_pass1_compaction_stops_once_within_char_and_byte_budgets(self) -> None:
        items = [
            {
                "role": "assistant",
                "full": f"Assistant:\n{'a' * 1000} #{i}",
                "compact": f"Assistant:\n{'a' * 200} #{i}",
            }
            for i in range(10)
        ]
        overhead = "System:\nrules"
        fitted = _fit_transcript_to_budget(items, overhead, max_prompt_chars=6500, max_prompt_bytes=6500)

        assembled = overhead + "\n\nConversation transcript:\n\n" + "\n\n".join(fitted)
        self.assertLessEqual(len(assembled), 6500)
        self.assertLessEqual(len(assembled.encode("utf-8")), 6500)
        self.assertEqual(len(fitted), 10)
        # Oldest items compacted first; newest items stay full once budget is met.
        self.assertEqual(fitted[0], items[0]["compact"])
        self.assertEqual(fitted[-1], items[-1]["full"])

    def test_multibyte_utf8_enforces_byte_budget_and_pins_latest_user(self) -> None:
        items = [
            {
                "role": "assistant",
                "full": f"Assistant:\n{'あ' * 600} #{i}",
                "compact": f"Assistant:\n{'あ' * 500} #{i}",
            }
            for i in range(25)
        ]
        items[5] = {
            "role": "user",
            "full": "User:\n最新のユーザー要求を保持する",
            "compact": "User:\n最新のユーザー要求を保持する",
        }
        overhead = "System:\nガイド"
        fitted = _fit_transcript_to_budget(items, overhead, max_prompt_chars=50000, max_prompt_bytes=12000)

        assembled = overhead + "\n\nConversation transcript:\n\n" + "\n\n".join(fitted)
        self.assertLessEqual(len(assembled.encode("utf-8")), 12000)
        self.assertTrue(any("最新のユーザー要求を保持する" in entry for entry in fitted))
        self.assertTrue(any("omitted" in entry for entry in fitted))


if __name__ == "__main__":
    unittest.main()
