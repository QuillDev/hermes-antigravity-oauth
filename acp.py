"""Agent Client Protocol (ACP) transport for Antigravity (``agy acp``).

Drives ``agy acp`` over stdin/stdout JSON-RPC 2.0 NDJSON. Unlike ``agy``
stream-json mode (which only accepts text content blocks), ``agy acp`` natively
supports multimodal ``image`` content blocks, streaming ``agent_thought_chunk``
reasoning deltas, multi-turn prompt caching, and per-turn token usage.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Iterator

try:
    from . import native_tools as _native_tools
    from .alias import StreamUnmasker, mask as _mask, unmask
    from .process import _own_process_group
    from .prompt import _ToolCallStreamBuffer
    from .stream import _ready
except ImportError:
    import native_tools as _native_tools
    from alias import StreamUnmasker, mask as _mask, unmask
    from process import _own_process_group
    from prompt import _ToolCallStreamBuffer
    from stream import _ready

logger = logging.getLogger(__name__)

ACP_DEFAULT_ARGS: tuple[str, ...] = ("acp", "--include-default-customizations=false")
_ACP_PROTOCOL_VERSION = 1
_HANDSHAKE_TIMEOUT_SECONDS = 30.0
_OUTPUT_LIMIT_STOP_REASONS = frozenset({"max_tokens", "max_output_tokens", "length"})


def is_acp_args(args: list[str] | tuple[str, ...]) -> bool:
    """True when ``args`` launches ``agy acp`` rather than stream-json mode."""
    return bool(args) and str(args[0]).strip() == "acp"


def mask_prompt_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply alias masking to text blocks only; never mutate base64 image data."""
    masked: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            masked.append({"type": "text", "text": _mask(str(block.get("text") or ""))})
        else:
            masked.append(dict(block))
    return masked


def _format_rpc_error(err: Any) -> str:
    if isinstance(err, dict):
        msg = err.get("message")
        code = err.get("code")
        if msg and code is not None:
            return f"{msg} (code {code})"
        if msg:
            return str(msg)
        return json.dumps(err, ensure_ascii=False)
    return str(err or "Unknown ACP error")


def _usage_from_acp(raw: dict[str, Any] | None) -> SimpleNamespace:
    """Convert ACP ``session/prompt`` result usage into OpenAI ChatCompletion usage.

    ``agy acp`` reports uncached input tokens in ``inputTokens``, cached prompt
    tokens in ``cachedReadTokens``, and total output (including thoughts) in
    ``outputTokens`` (``inputTokens + cachedReadTokens + outputTokens == totalTokens``).
    OpenAI's ``prompt_tokens`` includes cached tokens (``cached_tokens <= prompt_tokens``).
    """
    data = raw if isinstance(raw, dict) else {}
    uncached_input = int(data.get("inputTokens") or data.get("input_tokens") or 0)
    cached_tokens = int(data.get("cachedReadTokens") or data.get("cached_read_tokens") or 0)
    prompt_tokens = uncached_input + cached_tokens
    completion_tokens = int(data.get("outputTokens") or data.get("output_tokens") or 0)
    total_tokens = int(
        data.get("totalTokens")
        or data.get("total_tokens")
        or (prompt_tokens + completion_tokens)
    )
    raw_thoughts = data.get("thoughtTokens")
    if raw_thoughts is None:
        raw_thoughts = data.get("thought_tokens")
    thought_tokens = int(raw_thoughts) if raw_thoughts is not None else None
    return SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        completion_tokens_details=(
            SimpleNamespace(reasoning_tokens=thought_tokens)
            if thought_tokens is not None
            else None
        ),
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
    )


class ACPWorker:
    """Manages one ``agy acp`` subprocess and its single stdout/stderr readers."""

    def __init__(
        self,
        *,
        proc: subprocess.Popen,
        cwd: str,
        model: str,
        effort: str | None,
        terminate_fn: Any,
        handshake_timeout: float = _HANDSHAKE_TIMEOUT_SECONDS,
    ) -> None:
        self.proc = proc
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self._terminate_fn = terminate_fn
        self._handshake_timeout = handshake_timeout

        self.session_id: str = ""
        self.incoming: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._write_lock = threading.Lock()
        self._req_id = 0
        self._closed = False

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
            name="agy-acp-stderr",
        )
        self._stderr_thread.start()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            daemon=True,
            name="agy-acp-stdout",
        )
        self._stdout_thread.start()

    def _read_stdout(self) -> None:
        stdout = self.proc.stdout
        if stdout is None:
            self.incoming.put(None)
            return
        try:
            while True:
                raw_line = stdout.readline()
                if not raw_line:
                    break
                line = raw_line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    self.incoming.put(obj)
        except Exception:
            pass
        finally:
            self.incoming.put(None)

    def _drain_stderr(self) -> None:
        stderr = self.proc.stderr
        if stderr is None:
            return
        try:
            for line in stderr:
                self._stderr_lines.append(line.rstrip("\r\n"))
                if len(self._stderr_lines) > 200:
                    self._stderr_lines = self._stderr_lines[-200:]
        except Exception:
            pass

    def get_stderr(self) -> str:
        return "\n".join(self._stderr_lines).strip()

    def next_request_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def send_json(self, obj: dict[str, Any]) -> None:
        with self._write_lock:
            if self.proc.stdin is None or self.proc.poll() is not None:
                raise BrokenPipeError("Antigravity ACP process is not running")
            self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()

    def respond_permission_cancelled(self, rpc_id: Any) -> None:
        with contextlib.suppress(Exception):
            self.send_json({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"outcome": {"outcome": "cancelled"}},
            })

    def respond_method_not_found(self, rpc_id: Any, method: str) -> None:
        with contextlib.suppress(Exception):
            self.send_json({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not supported: {method}"},
            })

    def cancel_turn(self) -> None:
        if not self.session_id:
            return
        with contextlib.suppress(Exception):
            self.send_json({
                "jsonrpc": "2.0",
                "method": "session/cancel",
                "params": {"sessionId": self.session_id},
            })

    def _rpc_call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        req_id = self.next_request_id()
        self.send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })
        deadline = time.monotonic() + self._handshake_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Antigravity ACP {method} timed out after {int(self._handshake_timeout)}s"
                )
            try:
                msg = self.incoming.get(timeout=min(0.25, remaining))
            except queue.Empty:
                continue
            if msg is None:
                stderr = self.get_stderr()
                detail = f": {stderr}" if stderr else ""
                raise RuntimeError(f"Antigravity ACP process exited during {method}{detail}")
            if "method" in msg and "id" in msg:
                if msg.get("method") == "session/request_permission":
                    self.respond_permission_cancelled(msg.get("id"))
                else:
                    self.respond_method_not_found(msg.get("id"), str(msg.get("method") or ""))
                continue
            if msg.get("id") == req_id and ("result" in msg or "error" in msg):
                if "error" in msg:
                    raise RuntimeError(
                        f"Antigravity model error: {_format_rpc_error(msg['error'])}"
                    )
                res = msg.get("result")
                return res if isinstance(res, dict) else {}

    def handshake(self) -> None:
        self._rpc_call(
            "initialize",
            {
                "protocolVersion": _ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {"name": "antigravity-oauth", "version": "0.2.0"},
            },
        )
        sess = self._rpc_call(
            "session/new",
            {
                "cwd": self.cwd,
                "mcpServers": [],
            },
        )
        sid = sess.get("sessionId")
        if not sid:
            raise RuntimeError("Antigravity ACP session/new returned no sessionId")
        self.session_id = str(sid)

        if self.model:
            self._rpc_call(
                "session/set_model",
                {"sessionId": self.session_id, "modelId": self.model},
            )

        if self.effort:
            try:
                self._rpc_call(
                    "session/set_config_option",
                    {
                        "sessionId": self.session_id,
                        "configId": "thought_level",
                        "value": self.effort,
                    },
                )
            except RuntimeError as exc:
                logger.debug(
                    "Ignoring ACP thought_level=%s rejection for model %s: %s",
                    self.effort,
                    self.model,
                    exc,
                )

    def send_prompt(self, prompt_blocks: list[dict[str, Any]]) -> int:
        if not self.session_id:
            raise RuntimeError("Antigravity ACP session is not initialized")
        masked = mask_prompt_blocks(prompt_blocks)
        if not masked:
            masked = [{"type": "text", "text": ""}]
        req_id = self.next_request_id()
        self.send_json({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "session/prompt",
            "params": {
                "sessionId": self.session_id,
                "prompt": masked,
            },
        })
        return req_id

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            if self.proc.stdin:
                self.proc.stdin.close()
        self._terminate_fn(self.proc)
        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=0.3)


def spawn_acp_worker(
    *,
    command: str,
    args: list[str],
    cwd: str,
    env: dict[str, str],
    model: str,
    effort: str | None,
    terminate_fn: Any,
    popen_fn: Any = None,
) -> ACPWorker:
    """Spawn ``agy acp``, perform ACP handshake, and return a ready :class:`ACPWorker`."""
    cmd_args = [command, *args]
    popen = popen_fn or subprocess.Popen
    proc = popen(
        cmd_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        cwd=cwd,
        env=env,
        **_own_process_group(),
    )
    worker = ACPWorker(
        proc=proc,
        cwd=cwd,
        model=model,
        effort=effort,
        terminate_fn=terminate_fn,
    )
    try:
        worker.handshake()
    except Exception:
        worker.close()
        raise
    return worker


class AntigravityACPStream(Iterator[Any]):
    """Streaming iterator yielding OpenAI ChatCompletionChunk objects from ``agy acp``.

    Mirrors :class:`stream.AntigravityStream`'s sync/async/awaitable contract so
    Hermes' main loop, auxiliary async path, and Relay all treat it identically.
    """

    response: Any = None

    def __init__(
        self,
        *,
        worker: ACPWorker,
        req_id: int,
        client: Any,
        model: str,
        timeout: float,
        tools: list[dict[str, Any]] | None = None,
        is_worker: bool = False,
        worker_lock: threading.Lock | None = None,
        worker_lock_held: bool | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if (worker_lock is None) != (worker_lock_held is None):
            raise ValueError("worker_lock and worker_lock_held must be passed together")
        self.worker = worker
        self.proc = worker.proc
        self.req_id = req_id
        self.client = client
        self.model = model
        self.timeout = timeout
        self.has_tools = bool(tools)
        self.tool_names = _native_tools.tool_names(tools)
        self.is_worker = is_worker
        self.worker_lock = worker_lock
        self._worker_lock_held = bool(worker_lock_held and worker_lock is not None)
        self.messages = messages
        self.conversation_id = worker.session_id

        self._content_alias = StreamUnmasker()
        self._reasoning_alias = StreamUnmasker()
        self._closed = False
        self._finished = False
        self._interrupted = False
        self._async_executor: ThreadPoolExecutor | None = None
        self._generator = self._stream_generator()

    def __iter__(self) -> "AntigravityACPStream":
        return self

    def __await__(self) -> Any:
        return _ready(self).__await__()

    def __aiter__(self) -> "AntigravityACPStream":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            return next(self._generator)
        except StopIteration:
            self._retire_executor()
            raise
        except Exception:
            self.close()
            raise

    def _pull_chunk(self) -> tuple[bool, Any]:
        try:
            return False, self.__next__()
        except StopIteration:
            return True, None

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        executor = self._async_executor
        if executor is None:
            executor = self._async_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="agy-acp-stream"
            )
        try:
            exhausted, chunk = await asyncio.get_running_loop().run_in_executor(
                executor, self._pull_chunk
            )
        except asyncio.CancelledError:
            self.close()
            raise
        if exhausted:
            raise StopAsyncIteration
        return chunk

    async def aclose(self) -> None:
        self.close()

    def _retire_executor(self) -> None:
        executor = self._async_executor
        if executor is not None:
            self._async_executor = None
            executor.shutdown(wait=False)

    def _release_worker_lock_once(self) -> None:
        if self._worker_lock_held and self.worker_lock is not None:
            self._worker_lock_held = False
            with contextlib.suppress(RuntimeError):
                self.worker_lock.release()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.is_worker:
            if not self._finished and self.proc is self.client._worker_proc:
                self._interrupted = True
                self.worker.cancel_turn()
                self.client._terminate_worker()
            self._release_worker_lock_once()
        else:
            self.worker.cancel_turn()
            with self.client._lock:
                self.client._active_processes.discard(self.proc)
            self.worker.close()
        self._retire_executor()

    def _make_chunk(
        self,
        *,
        content: str | None = None,
        tool_calls: list[Any] | None = None,
        finish_reason: str | None = None,
        reasoning_content: str | None = None,
    ) -> Any:
        content = self._content_alias.feed(content) if content else content
        reasoning_content = (
            self._reasoning_alias.feed(reasoning_content)
            if reasoning_content
            else reasoning_content
        )
        if tool_calls or finish_reason:
            content = (content or "") + self._content_alias.flush() or None
            reasoning_content = (reasoning_content or "") + self._reasoning_alias.flush() or None
        for call in tool_calls or ():
            call.function.name = unmask(call.function.name)
            call.function.arguments = unmask(call.function.arguments)
        delta = SimpleNamespace(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning=None,
            reasoning_content=reasoning_content,
        )
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
        return SimpleNamespace(
            id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
            choices=[choice],
            model=self.model,
            usage=None,
        )

    def _translate_acp_tool_call(self, tool_call: dict[str, Any]) -> Any | None:
        if not self.has_tools or not isinstance(tool_call, dict):
            return None
        native_name = str(tool_call.get("name") or "").strip()
        params = tool_call.get("rawInput")
        if not isinstance(params, dict):
            params = tool_call.get("args")
        if not native_name or not isinstance(params, dict):
            return None
        step = {
            "tool_name": native_name,
            "tool_info": {"name": native_name, "parameters": params},
            "step_index": 0,
        }
        return _native_tools.translate(step, self.tool_names, index=0)

    def _stream_generator(self) -> Iterator[Any]:
        deadline = time.monotonic() + self.timeout
        tool_parser = _ToolCallStreamBuffer(has_tools=self.has_tools)
        usage_raw: dict[str, Any] = {}
        neutralized_tool_step = False
        output_truncated = False
        saw_result = False
        success = False

        try:
            while time.monotonic() < deadline:
                if self._closed:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    msg = self.worker.incoming.get(timeout=min(0.25, remaining))
                except queue.Empty:
                    continue

                if msg is None:
                    break

                if "method" in msg and "id" in msg:
                    method = str(msg.get("method") or "")
                    if method == "session/request_permission":
                        self.worker.respond_permission_cancelled(msg.get("id"))
                        tc_obj = (msg.get("params") or {}).get("toolCall")
                        if not tool_parser.has_tool_calls and isinstance(tc_obj, dict):
                            translated = self._translate_acp_tool_call(tc_obj)
                            if translated is not None:
                                logger.info(
                                    "Antigravity ACP native tool '%s' re-issued as Hermes tool '%s'.",
                                    tc_obj.get("name"),
                                    translated.function.name,
                                )
                                tool_parser.has_tool_calls = True
                                tool_parser.clear()
                                yield self._make_chunk(tool_calls=[translated])
                                neutralized_tool_step = True
                                self.close()
                                break
                            logger.warning(
                                "Antigravity ACP attempted native tool '%s'; permission denied to prevent host execution.",
                                tc_obj.get("name"),
                            )
                    else:
                        self.worker.respond_method_not_found(msg.get("id"), method)
                    continue

                if msg.get("id") == self.req_id and ("result" in msg or "error" in msg):
                    if "error" in msg:
                        raise RuntimeError(
                            f"Antigravity model error: {_format_rpc_error(msg['error'])}"
                        )
                    res = msg.get("result") or {}
                    saw_result = True
                    if isinstance(res, dict):
                        if isinstance(res.get("usage"), dict):
                            usage_raw = res["usage"]
                        stop_reason = str(res.get("stopReason") or "").strip().lower()
                        if stop_reason in _OUTPUT_LIMIT_STOP_REASONS:
                            output_truncated = True
                    success = not output_truncated and bool(
                        tool_parser.has_content or tool_parser.has_tool_calls or tool_parser.has_buffered
                    )
                    break

                if msg.get("method") != "session/update":
                    continue
                params = msg.get("params")
                if not isinstance(params, dict):
                    continue
                update = params.get("update")
                if not isinstance(update, dict):
                    continue
                kind = update.get("sessionUpdate")

                if kind == "agent_thought_chunk":
                    content = update.get("content")
                    thought = content.get("text") if isinstance(content, dict) else ""
                    if isinstance(thought, str) and thought:
                        yield self._make_chunk(reasoning_content=thought)
                    continue

                if kind == "agent_message_chunk":
                    content = update.get("content")
                    text_delta = content.get("text") if isinstance(content, dict) else ""
                    if not isinstance(text_delta, str) or not text_delta:
                        continue
                    for event_kind, payload in tool_parser.feed(text_delta):
                        if event_kind == "tool_call":
                            yield self._make_chunk(tool_calls=[payload])
                        else:
                            yield self._make_chunk(content=payload)

            for event_kind, payload in tool_parser.flush():
                if event_kind == "tool_call":
                    yield self._make_chunk(tool_calls=[payload])
                else:
                    yield self._make_chunk(content=payload)

            if not neutralized_tool_step and not saw_result:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Antigravity ACP stream timed out after {int(self.timeout)}s"
                    )
                stderr = self.worker.get_stderr()
                exit_code = self.proc.poll()
                detail = f"process exited with return code {exit_code}" if exit_code is not None else "no result received"
                if stderr:
                    detail += f"; stderr: {stderr}"
                raise RuntimeError(f"Antigravity execution failed: {detail}")

            if not neutralized_tool_step and not success and not output_truncated:
                raise RuntimeError("Antigravity execution failed: empty result (status='SUCCESS')")

            if output_truncated:
                finish_reason = "length"
            else:
                finish_reason = "tool_calls" if tool_parser.has_tool_calls else "stop"

            yield self._make_chunk(finish_reason=finish_reason)
            yield SimpleNamespace(
                id=self.conversation_id or f"agy-{int(time.time() * 1000)}",
                choices=[],
                model=self.model,
                usage=_usage_from_acp(usage_raw),
            )
            self._finished = True
        except Exception:
            success = False
            raise
        finally:
            if self.is_worker:
                if output_truncated and not success and self.proc is self.client._worker_proc:
                    self.client._terminate_worker()
                if success and not self._interrupted:
                    if self.proc is self.client._worker_proc:
                        self.client._worker_conversation_id = self.conversation_id
                    self.client._update_worker_history(self.messages)
                    self._release_worker_lock_once()
                else:
                    self.close()
            else:
                self.close()
