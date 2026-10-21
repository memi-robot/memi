"""The Plan client `LLMClient`, OpenAI chat completions or Claude messages over the standard library.

    client = LLMClient("gpt-6-astra", effort="low", max_tokens=budget("low"), timeout=1800.0)
    response = await client.complete(system=..., user=..., purpose="BinFill.optimize")
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DEFAULT_URL = "https://api.openai.com/v1/chat/completions"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_VERSION = "2023-06-01"
EFFORTS = ["none", "low", "medium", "high", "max", "xhigh"]

#: Attempts after the first, for a rate limit, a server error or a dropped connection.
RETRIES = 30
#: The longest wait between two attempts, in seconds, before jitter.
BACKOFF_CAP = 120

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


class CallFailed(RuntimeError):
    """An LLM call failed after all its retries.

    This is a settings problem (key, model name, quota, endpoint), not a policy
    outcome. The run stops where it failed, and a rerun with fixed settings
    resumes from there.
    """


def budget(effort: str) -> int:
    """The answer budget a reasoning model needs at this effort.

    OpenAI counts reasoning against `max_completion_tokens`, so a reasoning
    model given a small budget thinks until it is gone and never answers.
    `none` spends nothing on reasoning, so its budget is the answer alone.
    """
    return {"none": 8192, "low": 32768}.get(effort, 65536)


@dataclass(frozen=True)
class LLMResponse:
    """One assistant answer."""

    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    raw: Any = None
    #: The model's own reasoning, where the host returns it.
    reasoning: str = ""


def is_claude(model: str) -> bool:
    """Whether `model` is served by the Claude messages API."""
    return model.startswith("claude-")


def ca_file() -> str | None:
    """The CA bundle to verify with. SSL_CERT_FILE or REQUESTS_CA_BUNDLE, else
    the system bundle when there is one."""
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    system_ca = Path("/etc/ssl/certs/ca-certificates.crt")
    if not cafile and system_ca.exists():
        cafile = str(system_ca)
    return cafile


def default_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    """One POST, the response body as bytes."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    context = ssl.create_default_context(cafile=ca_file())
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        return resp.read()


class JSONLTrace:
    """One row per call, appended to a JSONL file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        #: The 1-based index of each call, in the order the calls start.
        self.indices = itertools.count(1)

    def append(self, *, purpose: str, call_index: int, model: str, latency: float,
               usage: dict[str, Any] | None = None, response: str = "",
               reasoning: str = "", error: str = "") -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"purpose": purpose, "call_index": call_index, "model": model,
               "latency": latency, "usage": usage or {}, "response": response,
               "reasoning": reasoning, "error": error}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def cache_key(purpose: str) -> str:
    """The prompt cache key of a call, from its purpose.

    A stable key per task and role routes the calls that share a prompt prefix
    to the same cache shard. The episode, step and run segments (`ep3`, `t40`,
    `run2`) would make every call its own key, so they are dropped.
    """
    stable = [p for p in purpose.split(".")
              if p and not re.fullmatch(r"(?:ep|t|run)\d+", p)]
    return ":".join(stable)[:128]


class LLMClient:
    """One model at one reasoning effort. The answer is always asked for as one JSON object.

    The API key is OPENAI_API_KEY and the endpoint OPENAI_API_URL, the public
    endpoint when unset. A `claude-*` model goes to the Claude messages API,
    with ANTHROPIC_API_KEY and ANTHROPIC_API_URL.
    """

    def __init__(self, model: str, *, effort: str, max_tokens: int, timeout: float,
                 trace_path: str | Path | None = None, transport: Transport | None = None) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = "anthropic" if is_claude(model) else "openai"
        if is_claude(model):
            env_key, env_url, default_url = "ANTHROPIC_API_KEY", "ANTHROPIC_API_URL", CLAUDE_URL
        else:
            env_key, env_url, default_url = "OPENAI_API_KEY", "OPENAI_API_URL", DEFAULT_URL
        self.api_key = os.environ.get(env_key, "")
        self.url = os.environ.get(env_url) or default_url
        self.transport = transport or (claude_transport if self.provider == "anthropic"
                                       else default_transport)
        self.trace = JSONLTrace(trace_path)

    async def complete(self, *, system: str, user: str, purpose: str) -> LLMResponse:
        """One user turn."""
        return await self.call(system=system,
                               messages=[{"role": "user", "content": user}],
                               purpose=purpose)

    async def call(self, *, system: str, messages: list[dict[str, Any]],
                   purpose: str) -> LLMResponse:
        """One request with `messages` after the system turn, retried, with one row in the trace file.

        Every Plan call reaches its model here. `memi.plan.replay` patches it
        to answer from the released traces.
        """
        if not self.api_key:
            name = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
            raise RuntimeError(f"{name} is required. Put it in .env or the environment.")
        call_index = next(self.trace.indices)
        started = time.monotonic()
        response_text, reasoning, error = "", "", ""
        usage: dict[str, Any] = {}
        try:
            wire = [{"role": "system", "content": system}] + list(messages)
            data = await self._request(self._payload(wire, purpose))
            parsed = json.loads(data.decode("utf-8"))
            usage = dict(parsed.get("usage") or {})
            response_text = _text(parsed)
            reasoning = _reasoning(parsed)
            # An empty answer that stopped on `length` spent the whole budget on
            # reasoning. The trace row says so, so it does not read as a model
            # that answered nothing.
            if not response_text:
                reason = ((parsed.get("choices") or [{}])[0]).get("finish_reason")
                if reason == "length":
                    spent = (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", usage.get("completion_tokens"))
                    error = (f"empty answer: all {spent} completion tokens went to "
                             f"reasoning and the budget ran out before the answer "
                             f"started (max_tokens={self.max_tokens})")
                    print(f"[llm] {purpose}: {error}", flush=True)
            return LLMResponse(text=response_text, usage=usage, raw=parsed,
                               reasoning=reasoning)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.trace.append(purpose=purpose, call_index=call_index, model=self.model,
                              latency=time.monotonic() - started, usage=usage,
                              response=response_text, reasoning=reasoning, error=error)

    def _payload(self, messages: list[dict[str, Any]], purpose: str = "") -> dict[str, Any]:
        """The request body for `messages`, system turn included."""
        if self.provider == "anthropic":
            return self.claude_payload(messages)
        out: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_tokens,
            "reasoning_effort": self.effort,
        }
        key = cache_key(purpose)
        if key:
            out["prompt_cache_key"] = key
        return out

    def claude_payload(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """The Claude messages request body. The system turn becomes `system`.

        Claude has no JSON mode, so the prompts alone ask for the JSON object.
        The effort is Claude's own effort setting with adaptive thinking.
        """
        system = "\n\n".join(str(m["content"]) for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        out: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": turns,
            "stream": True,
        }
        if system:
            # Every call of a role shares this system prompt.
            out["system"] = [{"type": "text", "text": system,
                              "cache_control": {"type": "ephemeral"}}]
        if len(turns) > 1:
            # A conversation is cached up to its last turn, so its next call reads
            # the earlier turns. A single user turn is never sent twice, so caching
            # it would only pay the cache write.
            out["cache_control"] = {"type": "ephemeral"}
        if self.effort == "none":
            out["thinking"] = {"type": "disabled"}
        else:
            out["thinking"] = {"type": "adaptive", "display": "summarized"}
            out["output_config"] = {"effort": self.effort}
        return out

    async def _request(self, payload: dict[str, Any]) -> bytes:
        body = json.dumps(payload).encode("utf-8")
        if self.provider == "anthropic":
            headers = {"x-api-key": self.api_key, "anthropic-version": CLAUDE_VERSION,
                       "Content-Type": "application/json"}
        else:
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
        for attempt in range(RETRIES + 1):
            try:
                return await asyncio.to_thread(self.transport, self.url, headers, body,
                                               self.timeout)
            except urllib.error.HTTPError as exc:
                transient = exc.code in {408, 409, 429} or exc.code >= 500
                if not transient or attempt >= RETRIES:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt >= RETRIES:
                    raise
            # Nothing above here retries a call, so a rate limit has to clear
            # here or it ends the run. Jittered, because every call in an iteration
            # is sent at once and they all fail at once.
            delay = min(2 ** attempt, BACKOFF_CAP)
            await asyncio.sleep(delay * (0.5 + random.random()))


def _text(parsed: dict[str, Any]) -> str:
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) for p in content if p.get("type") == "text")
    return str(content)


def _reasoning(parsed: dict[str, Any]) -> str:
    """The reasoning the host returned, under `reasoning_content` or `reasoning`,
    as a string or as a list of summary parts. Empty when it returned none."""
    choices = parsed.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list):
            joined = "\n".join(str(part.get("text", part.get("summary", "")))
                               for part in value if isinstance(part, dict))
            if joined.strip():
                return joined
    return ""


#: Claude's stop reasons in the chat completions wording.
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
          "refusal": "content_filter"}


class ClaudeStreamBroken(OSError):
    """A Claude stream that is not a whole answer. An OSError, so it is retried
    like a dropped connection."""


def claude_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    """One streamed Claude request, returned as the JSON of a chat completion so the rest of the code sees one
    format."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    context = ssl.create_default_context(cafile=ca_file())
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        out, _ = read_claude_stream(resp, time.monotonic())
    return out


def read_claude_stream(lines, t0: float) -> tuple[bytes, dict]:
    """A Claude server-sent event stream, as the JSON of a chat completion, and its timing.

    The usage is put in the chat completions fields. `prompt_tokens` counts
    every input token, cached or not. `cached_tokens` are the cache reads and
    `cache_write_tokens` the cache writes. `reasoning_tokens` are the thinking
    tokens, which are part of `completion_tokens` as with OpenAI.
    Raises ClaudeStreamBroken on an error event, on a chunk that is not JSON,
    and on a stream that ends without `message_stop`.
    """
    first = first_text = None
    parts: list[str] = []
    thought: list[str] = []
    usage: dict[str, Any] = {}
    model, finish, done = "", None, False
    for raw in lines:
        line = (raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)).strip()
        if not line.startswith("data:"):
            continue
        if first is None:
            first = time.monotonic() - t0
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError as exc:
            raise ClaudeStreamBroken(f"unparseable stream chunk: {line[:200]!r}") from exc
        kind = event.get("type")
        if kind == "error":
            raise ClaudeStreamBroken(f"error event in the stream: "
                                     f"{json.dumps(event.get('error'))[:300]}")
        if kind == "message_start":
            message = event.get("message") or {}
            model = message.get("model", "")
            usage.update(message.get("usage") or {})
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                if first_text is None:
                    first_text = time.monotonic() - t0
                parts.append(delta.get("text", ""))
            elif delta.get("type") == "thinking_delta":
                thought.append(delta.get("thinking", ""))
        elif kind == "message_delta":
            finish = (event.get("delta") or {}).get("stop_reason") or finish
            usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
        elif kind == "message_stop":
            done = True
            break
    if not done:
        raise ClaudeStreamBroken("the stream ended without message_stop")
    read = int(usage.get("cache_read_input_tokens") or 0)
    write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    chat_usage = {
        "prompt_tokens": int(usage.get("input_tokens") or 0) + read + write,
        "completion_tokens": output,
        "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write},
        "completion_tokens_details": {"reasoning_tokens": int(
            (usage.get("output_tokens_details") or {}).get("thinking_tokens") or 0)},
    }
    message = {"content": "".join(parts)}
    if thought:
        message["reasoning_content"] = "".join(thought)
    timing = {"first_chunk_s": None if first is None else round(first, 3),
              "ttft_s": None if first_text is None else round(first_text, 3),
              "stream_s": round(time.monotonic() - t0, 3)}
    return json.dumps({"model": model, "usage": chat_usage, "choices": [
        {"message": message, "finish_reason": FINISH.get(finish, finish)}]}).encode("utf-8"), timing
