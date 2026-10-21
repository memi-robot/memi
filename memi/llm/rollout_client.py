"""The client of the Plan rollout policies, `send(system, messages, purpose) -> (text, record)`, streamed and timed."""
from __future__ import annotations

import asyncio
import json
import ssl
import time
import urllib.request

from .client import CallFailed, LLMClient, ca_file, is_claude, read_claude_stream


class StreamBroken(OSError):
    """The chat completions counterpart of ClaudeStreamBroken."""


#: Per-call usage fields. Missing ones stay 0.
TOKEN_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def usage_record(usage: dict) -> dict:
    """The chat completions usage block in the fields every record carries, and the cache writes when there are
    some. Cache writes are billed apart from the other input tokens."""
    u = usage or {}
    out = {
        "input_tokens": int(u.get("prompt_tokens") or 0),
        "cached_tokens": int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        "output_tokens": int(u.get("completion_tokens") or 0),
        "reasoning_tokens": int((u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                                or 0),
    }
    write = int((u.get("prompt_tokens_details") or {}).get("cache_write_tokens") or 0)
    if write:
        out["cache_write_tokens"] = write
    return out


class RolloutClient:
    """`LLMClient` with a streamed transport, so the first answer token is timed apart from the whole call.

    The payload, the prompt cache key, the timeout and the retries are the
    loop's. The model and effort are pinned, with no fallback to another model.
    """

    def __init__(self, model: str, effort: str, max_tokens: int, timeout: float):
        self.model, self.effort = model, effort
        self._attempts = 0
        self._timing: dict = {}

        def counting(url, headers, body, timeout_s):
            self._attempts += 1
            return self._streamed(url, headers, body, timeout_s)

        self.client = LLMClient(model, effort=effort, max_tokens=max_tokens,
                                   timeout=timeout, transport=counting)

    def _streamed(self, url: str, headers: dict, body: bytes, timeout: float) -> bytes:
        """One streamed request, returned as the JSON a plain request returns.

        The client parses the answer, the usage and the model id from that JSON
        exactly as it does without streaming. What streaming adds is timing,
        all measured on this client.
          first_chunk_s  time to the first streamed chunk
          ttft_s         time to the first token of the answer text. A reasoning
                         model reasons before it, so this includes the reasoning
          stream_s       time to the end of the stream
        """
        payload = json.loads(body)
        payload["stream"] = True
        if not is_claude(self.model):
            payload["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        t0 = time.monotonic()
        reader = read_claude_stream if is_claude(self.model) else read_stream
        with urllib.request.urlopen(req, timeout=timeout,
                                    context=ssl.create_default_context(cafile=ca_file())) as resp:
            out, timing = reader(resp, t0)
        self._timing = timing
        return out

    def send(self, system: str, messages: list[dict], purpose: str) -> tuple[str, dict]:
        """The answer to `messages`, the conversation after the system turn, and its usage record.

        A call that fails after its retries raises CallFailed and is never
        scored. An empty answer cut off by the token budget raises it at once.
        """
        self._attempts = 0
        self._timing = {}
        t0 = time.monotonic()
        rec = {"model": self.model, "effort": self.effort, "model_returned": "",
               **dict.fromkeys(TOKEN_FIELDS, 0), "error": ""}
        text = ""
        try:
            r = asyncio.run(self.client.call(system=system, messages=list(messages),
                                             purpose=purpose))
            text = r.text or ""
            rec.update(usage_record(r.usage))
            rec["model_returned"] = str((r.raw or {}).get("model", ""))
            rec["finish_reason"] = str(((r.raw or {}).get("choices") or [{}])[0]
                                       .get("finish_reason", ""))
            if r.reasoning:
                rec["reasoning"] = r.reasoning
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
            print(f"[rollout] {purpose}: call failed after {self._attempts} attempts: "
                  f"{rec['error']}", flush=True)
            raise CallFailed(rec["error"]) from exc
        if not text.strip() and rec["finish_reason"] == "length":
            msg = (f"empty answer cut off by the token budget (finish_reason=length, "
                   f"{rec['output_tokens']} output tokens, {rec['reasoning_tokens']} of them "
                   f"reasoning). Raise --max-tokens (now {self.client.max_tokens}) or "
                   f"lower the effort.")
            print(f"[rollout] {purpose}: {msg}", flush=True)
            raise CallFailed(msg)
        rec["latency"] = round(time.monotonic() - t0, 3)
        rec["retries"] = max(self._attempts - 1, 0)
        rec["streamed"] = True
        # Timing of the attempt that answered. Earlier failed attempts are in
        # `latency` only.
        rec.update(self._timing)
        return text, rec


def _round(x):
    return None if x is None else round(x, 3)


def read_stream(lines, t0: float) -> tuple[bytes, dict]:
    """A server-sent event stream, as the JSON a plain request returns, and its timing.

    Raises StreamBroken when the stream is not a whole answer. That is a reply
    with no `data:` events (such as plain JSON), an `error` event, a chunk that
    is not JSON, a stream that ends without [DONE], and a stream with no usage
    block. Reasoning text that the stream carries (`reasoning_content` or
    `reasoning` in the delta) is kept.
    """
    first = first_text = None
    parts: list[str] = []
    thought: list[str] = []
    usage: dict = {}
    model, finish = "", None
    events = done = 0
    for raw in lines:
        line = (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        events += 1
        if data == "[DONE]":
            done = 1
            break
        if first is None:
            first = time.monotonic() - t0
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as exc:
            raise StreamBroken(f"unparseable stream chunk: {data[:200]!r}") from exc
        if not isinstance(chunk, dict):
            raise StreamBroken(f"stream chunk is not an object: {data[:200]!r}")
        if chunk.get("error"):
            raise StreamBroken(f"error event in the stream: {json.dumps(chunk['error'])[:300]}")
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for c in chunk.get("choices") or []:
            delta = c.get("delta") or {}
            text = delta.get("content")
            if text:
                if first_text is None:
                    first_text = time.monotonic() - t0
                parts.append(text)
            for key in ("reasoning_content", "reasoning"):
                if isinstance(delta.get(key), str):
                    thought.append(delta[key])
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    if not events:
        raise StreamBroken("the reply to a stream request had no stream events "
                           "(plain JSON or an empty body)")
    if not done:
        raise StreamBroken("the stream ended without [DONE]")
    if not usage:
        raise StreamBroken("the stream carried no usage block (stream_options ignored)")
    message = {"content": "".join(parts)}
    if thought:
        message["reasoning_content"] = "".join(thought)
    timing = {"first_chunk_s": _round(first), "ttft_s": _round(first_text),
              "stream_s": _round(time.monotonic() - t0)}
    return json.dumps({"model": model, "usage": usage, "choices": [
        {"message": message, "finish_reason": finish}]}).encode("utf-8"), timing
