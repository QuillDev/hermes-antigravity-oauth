"""Behavior tests for Antigravity ACP transport and native multimodal vision."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

# Hermetic: keep provider discovery from loading an older installed copy from ~/.hermes/plugins
os.environ["HERMES_HOME"] = tempfile.mkdtemp(prefix="agy_acp_test_home_")

import alias as _alias
import __init__ as plugin
from acp import ACP_DEFAULT_ARGS
from client import AntigravityClient
from prompt import (
    _extract_image_block,
    _format_delta_acp_blocks,
    _format_messages_as_acp_blocks,
)

_TINY_PNG_BYTES = b"\x89PNG\r\n\x1a\nFAKEPNG"
_TINY_PNG_B64 = base64.b64encode(_TINY_PNG_BYTES).decode("ascii")
_TINY_PNG_URI = f"data:image/png;base64,{_TINY_PNG_B64}"
_TC_CLOSE = "</" + "tool_call>"


class _FakeACPProcess:
    """In-memory fake ``agy acp`` process speaking JSON-RPC 2.0 over stdio."""

    def __init__(
        self,
        *,
        turn_handler=None,
        set_effort_error: str | None = None,
        stderr_text: str = "",
    ) -> None:
        self._turn_handler = turn_handler
        self._set_effort_error = set_effort_error
        self._out: queue.Queue[str | None] = queue.Queue()
        self.stdout = self._Reader(self._out)
        self.stderr = io.StringIO(stderr_text)
        self.stdin = self._Writer(self)
        self.received: list[dict[str, Any]] = []
        self.server_replies: list[dict[str, Any]] = []
        self.permission_replied = threading.Event()
        self.session_id = "acp-session-1"
        self.pid = 31337
        self.terminated = False

    class _Reader:
        def __init__(self, q: queue.Queue[str | None]) -> None:
            self._q = q

        def readline(self) -> str:
            item = self._q.get()
            return "" if item is None else item

    class _Writer:
        def __init__(self, proc: _FakeACPProcess) -> None:
            self._proc = proc
            self._buf = ""

        def write(self, data: str) -> int:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line.strip():
                    self._proc._on_msg(json.loads(line))
            return len(data)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            self._proc._out.put(None)

    def emit(self, obj: dict[str, Any]) -> None:
        self._out.put(json.dumps(obj) + "\n")

    def _on_msg(self, msg: dict[str, Any]) -> None:
        self.received.append(msg)
        if "method" not in msg and "id" in msg:
            self.server_replies.append(msg)
            self.permission_replied.set()
            return
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        if method == "initialize":
            self.emit({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"protocolVersion": 1, "agentCapabilities": {"promptCapabilities": {"image": True}}},
            })
        elif method == "session/new":
            self.emit({"jsonrpc": "2.0", "id": req_id, "result": {"sessionId": self.session_id}})
        elif method == "session/set_model":
            self.emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/set_config_option":
            if self._set_effort_error:
                self.emit({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": self._set_effort_error},
                })
            else:
                self.emit({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            if self._turn_handler is not None:
                threading.Thread(
                    target=self._turn_handler,
                    args=(self, req_id, params),
                    daemon=True,
                ).start()
            else:
                self.emit({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": self.session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": "ok"},
                        },
                    },
                })
                self.emit({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 100, "outputTokens": 5, "cachedReadTokens": 0, "totalTokens": 105},
                    },
                })

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True
        self._out.put(None)

    def kill(self):
        self.terminated = True
        self._out.put(None)

    def wait(self, timeout=None):
        return 0


class ACPVisionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="agy_acp_test_")
        self.addCleanup(self._tmp.cleanup)
        self._env_patch = patch.dict(os.environ, {}, clear=False)
        self._env_patch.start()
        os.environ.pop("ANTIGRAVITY_TRANSPORT", None)
        self.addCleanup(self._env_patch.stop)

    def _make_client(self, **kwargs) -> AntigravityClient:
        client = AntigravityClient(cwd=self._tmp.name, command="agy", **kwargs)
        self.addCleanup(client.close)
        return client

    def test_empty_args_from_host_fall_back_to_acp(self):
        spawned: list[list[str]] = []

        def fake_popen(cmd, **kw):
            spawned.append(cmd)
            return _FakeACPProcess()

        with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=fake_popen):
            client = self._make_client(args=[])
            resp = client.chat.completions.create(
                model="gemini-4-neon-high",
                messages=[{"role": "user", "content": "hi"}],
            )
        self.assertEqual(resp.choices[0].message.content, "ok")
        self.assertEqual(spawned[0][:2], ["agy", "acp"])

    def test_profile_declares_acp_args_and_model_vision_capabilities(self):
        profile = plugin.antigravity_profile
        self.assertEqual(profile.process_args, ACP_DEFAULT_ARGS)
        self.assertNotIn("--dangerously-skip-permissions", profile.process_args)
        neon_caps = profile.model_capabilities.get("gemini-4-neon-high") or {}
        claude_caps = profile.model_capabilities.get("claude-sonnet-5-5-high") or {}
        gpt_caps = profile.model_capabilities.get("gpt-oss-120b-medium") or {}
        self.assertTrue(neon_caps["supports_vision"])
        self.assertEqual(neon_caps["context_window"], 1_000_000)
        self.assertTrue(claude_caps["supports_vision"])
        self.assertFalse(gpt_caps["supports_vision"])
        self.assertEqual(gpt_caps["context_window"], 131_072)
        self.assertEqual(profile.model_capabilities.get("unknown-vendor-model", {}), {})

        from agent.image_routing import decide_image_input_mode
        from tools.vision_tools import _native_tool_result_images

        self.assertEqual(decide_image_input_mode("antigravity-oauth", "gemini-4-neon-high", {}), "native")
        self.assertTrue(_native_tool_result_images("antigravity-oauth", "gemini-4-neon-high", {}))
        self.assertEqual(decide_image_input_mode("antigravity-oauth", "claude-sonnet-5-5-high", {}), "native")
        self.assertEqual(decide_image_input_mode("antigravity-oauth", "gpt-oss-120b-medium", {}), "text")
        self.assertFalse(_native_tool_result_images("antigravity-oauth", "gpt-oss-120b-medium", {}))

    def test_default_acp_sends_native_images_and_maps_usage_and_cache(self):
        spawned: list[tuple[list[str], _FakeACPProcess]] = []

        def handle_turn(proc: _FakeACPProcess, req_id: int, params: dict[str, Any]):
            imgs = [b for b in params["prompt"] if b.get("type") == "image"]
            proc.emit({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": proc.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"images={len(imgs)}:{imgs[0]['mimeType']}"},
                    },
                },
            })
            proc.emit({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "stopReason": "end_turn",
                    "usage": {
                        "inputTokens": 300,
                        "cachedReadTokens": 700,
                        "outputTokens": 50,
                        "thoughtTokens": 30,
                        "totalTokens": 1050,
                    },
                },
            })

        def fake_popen(cmd, **kw):
            p = _FakeACPProcess(turn_handler=handle_turn)
            spawned.append((cmd, p))
            return p

        with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=fake_popen):
            client = self._make_client()
            resp = client.chat.completions.create(
                model="gemini-4-neon-high",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this?"},
                        {"type": "image_url", "image_url": {"url": _TINY_PNG_URI}},
                    ],
                }],
            )
        self.assertEqual(spawned[0][0][:2], ["agy", "acp"])
        self.assertEqual(resp.choices[0].message.content, "images=1:image/png")
        self.assertEqual(resp.usage.prompt_tokens, 1000)
        self.assertEqual(resp.usage.completion_tokens, 50)
        self.assertEqual(resp.usage.total_tokens, 1050)
        self.assertEqual(resp.usage.prompt_tokens_details.cached_tokens, 700)
        self.assertEqual(resp.usage.completion_tokens_details.reasoning_tokens, 30)
        prompt_reqs = [m for m in spawned[0][1].received if m.get("method") == "session/prompt"]
        self.assertEqual(
            [b for b in prompt_reqs[0]["params"]["prompt"] if b.get("type") == "image"],
            [{"type": "image", "mimeType": "image/png", "data": _TINY_PNG_B64}],
        )

    def test_persistent_acp_reuses_worker_and_forwards_multimodal_tool_result(self):
        seen: list[list[dict[str, Any]]] = []
        spawned: list[_FakeACPProcess] = []

        def handle_turn(proc: _FakeACPProcess, req_id: int, params: dict[str, Any]):
            seen.append(params["prompt"])
            n = len(seen)
            proc.emit({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": proc.session_id,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"turn-{n}"},
                    },
                },
            })
            proc.emit({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"stopReason": "end_turn", "usage": {"inputTokens": 50, "outputTokens": 3, "totalTokens": 53}},
            })

        def fake_popen(cmd, **kw):
            p = _FakeACPProcess(turn_handler=handle_turn)
            spawned.append(p)
            return p

        with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=fake_popen):
            client = self._make_client()
            m1 = [{"role": "user", "content": "Load /tmp/x.png"}]
            r1 = client.chat.completions.create(model="gemini-4-neon-high", messages=m1)
            self.assertEqual(r1.choices[0].message.content, "turn-1")

            m2 = m1 + [
                {"role": "assistant", "content": "turn-1"},
                {
                    "role": "tool",
                    "tool_call_id": "call_v1",
                    "content": {
                        "_multimodal": True,
                        "content": [
                            {"type": "text", "text": "Image loaded into your context"},
                            {"type": "image_url", "image_url": {"url": _TINY_PNG_URI}},
                        ],
                    },
                },
            ]
            r2 = client.chat.completions.create(model="gemini-4-neon-high", messages=m2)
            self.assertEqual(r2.choices[0].message.content, "turn-2")
        self.assertEqual(len(spawned), 1)
        self.assertEqual(len(seen), 2)
        self.assertEqual([b for b in seen[1] if b.get("type") == "image"], [{"type": "image", "mimeType": "image/png", "data": _TINY_PNG_B64}])

    def test_acp_streams_thoughts_parses_tool_calls_and_unmasks_alias(self):
        captured: list[list[dict[str, Any]]] = []

        def handle_turn(proc: _FakeACPProcess, req_id: int, params: dict[str, Any]):
            captured.append(params["prompt"])
            proc.emit({
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": proc.session_id,
                    "update": {
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": _alias.mask("thinking for hermes")},
                    },
                },
            })
            payload = json.dumps({
                "id": "tc_1",
                "type": "function",
                "function": {"name": "ping", "arguments": json.dumps({"host": "1.1.1.1"})},
            })
            for piece in ["Checking <tool", "_call>" + payload + _TC_CLOSE]:
                proc.emit({
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "sessionId": proc.session_id,
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": piece},
                        },
                    },
                })
            proc.emit({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"stopReason": "end_turn", "usage": {"inputTokens": 20, "outputTokens": 10, "totalTokens": 30}},
            })

        tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]
        with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=lambda *a, **kw: _FakeACPProcess(turn_handler=handle_turn)):
            client = self._make_client()
            resp = client.chat.completions.create(
                model="gemini-4-neon-high",
                tools=tools,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Ping for hermes"},
                        {"type": "image_url", "image_url": {"url": _TINY_PNG_URI}},
                    ],
                }],
            )
            awaited = asyncio.run(self._await_val(resp))
        self.assertIs(awaited, resp)
        self.assertEqual(resp.choices[0].message.reasoning_content, "thinking for hermes")
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        self.assertEqual(resp.choices[0].message.tool_calls[0].function.name, "ping")
        self.assertEqual(json.loads(resp.choices[0].message.tool_calls[0].function.arguments), {"host": "1.1.1.1"})
        self.assertEqual([b for b in captured[0] if b.get("type") == "image"][0]["data"], _TINY_PNG_B64)

    @staticmethod
    async def _await_val(val):
        return await val

    def test_acp_permission_request_cancelled_and_translated_to_host_tool(self):
        spawned: list[_FakeACPProcess] = []

        def handle_turn(proc: _FakeACPProcess, req_id: int, params: dict[str, Any]):
            proc.emit({
                "jsonrpc": "2.0",
                "id": "perm-1",
                "method": "session/request_permission",
                "params": {
                    "sessionId": proc.session_id,
                    "toolCall": {
                        "name": "run_command",
                        "rawInput": {"CommandLine": "echo hi"},
                    },
                },
            })
            proc.permission_replied.wait(timeout=3.0)

        def fake_popen(cmd, **kw):
            p = _FakeACPProcess(turn_handler=handle_turn)
            spawned.append(p)
            return p

        tools = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]
        with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=fake_popen):
            client = self._make_client()
            resp = client.chat.completions.create(
                model="claude-sonnet-5-5-low",
                tools=tools,
                messages=[{"role": "user", "content": "Run echo hi"}],
            )
        self.assertEqual(resp.choices[0].finish_reason, "tool_calls")
        tc = resp.choices[0].message.tool_calls[0]
        self.assertEqual(tc.function.name, "terminal")
        self.assertEqual(json.loads(tc.function.arguments), {"command": "echo hi"})
        self.assertEqual(
            spawned[0].server_replies,
            [{"jsonrpc": "2.0", "id": "perm-1", "result": {"outcome": {"outcome": "cancelled"}}}],
        )

    def test_acp_async_streaming_and_effort_rejection_tolerance(self):
        def fake_popen(cmd, **kw):
            return _FakeACPProcess(set_effort_error="Invalid thought_level")

        async def consume():
            with patch("client.is_authenticated", return_value=True), patch("subprocess.Popen", side_effect=fake_popen):
                client = self._make_client()
                stream = await client.chat.completions.create(
                    model="gemini-4-neon",
                    reasoning_effort="high",
                    stream=True,
                    messages=[{"role": "user", "content": "hi"}],
                )
                chunks = []
                async for chunk in stream:
                    chunks.append(chunk)
                return chunks

        chunks = asyncio.run(consume())
        text = "".join(c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content)
        self.assertEqual(text, "ok")

    def test_prompt_image_extraction_formats(self):
        img_path = Path(self._tmp.name) / "dot.png"
        img_path.write_bytes(_TINY_PNG_BYTES)

        self.assertEqual(_extract_image_block({"type": "image_url", "image_url": {"url": _TINY_PNG_URI}}), {
            "type": "image", "mimeType": "image/png", "data": _TINY_PNG_B64,
        })
        self.assertEqual(_extract_image_block({"type": "input_image", "image_url": _TINY_PNG_URI}), {
            "type": "image", "mimeType": "image/png", "data": _TINY_PNG_B64,
        })
        self.assertEqual(_extract_image_block({
            "type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": "QUJD"},
        }), {"type": "image", "mimeType": "image/webp", "data": "QUJD"})
        local_blk = _extract_image_block({"type": "image_url", "image_url": {"url": f"file://{img_path}"}})
        self.assertEqual(local_blk, {"type": "image", "mimeType": "image/png", "data": _TINY_PNG_B64})

        blocks = _format_delta_acp_blocks([{
            "role": "user",
            "content": [
                {"type": "text", "text": "See"},
                {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
            ],
        }])
        self.assertEqual(len(blocks), 1)
        self.assertIn("https://example.com/pic.png", blocks[0]["text"])

        img_only = _format_messages_as_acp_blocks([{
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _TINY_PNG_URI}}],
        }])
        self.assertEqual(len(img_only), 2)
        self.assertEqual(img_only[0]["type"], "text")
        self.assertEqual(img_only[1]["type"], "image")


if __name__ == "__main__":
    unittest.main()
