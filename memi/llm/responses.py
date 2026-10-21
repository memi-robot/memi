"""The Act client, OpenAI Responses over the standard library, with one cost ledger shared by every process.

    client = Client(config.ASTRA, tag="BinFill")
    reply = client.ask(system, content, purpose="BinFill.policy.forward", prefix=shared)

Every call appends its tokens and cost to .cache/e2e/ledger.jsonl, which concurrent processes share through a shared
file system. `spent()` sums it.
"""
from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request

from ..act import config
from ..paths import load_env
from .client import CallFailed, ca_file
from .parse import parse_object
from .prices import PRICES, dollars

LEDGER = config.CACHE / "ledger.jsonl"
PREFIX_IDS = config.CACHE / "prefix_ids.json"
URL = "https://api.openai.com/v1/responses"
RETRIED_HTTP = (408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529)
ATTEMPTS = 15                  # about 13 minutes. A shared rate limit clears within a minute or two.
TIMEOUT = 600.0                # seconds per request


def api_key() -> str:
    load_env()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise CallFailed("OPENAI_API_KEY is required. Put it in .env or the environment.")
    return key


def spent() -> float:
    """Dollars recorded in the ledger so far."""
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text().splitlines():
        try:
            total += float(json.loads(line).get("cost", 0))
        except (ValueError, json.JSONDecodeError):
            continue
    return total


def cost_of(usage: dict, model: str) -> float:
    det = usage.get("input_tokens_details") or {}
    return dollars(model, usage.get("input_tokens") or 0, det.get("cached_tokens") or 0,
                   det.get("cache_write_tokens") or 0, usage.get("output_tokens") or 0)


def log_call(usage: dict, model: str, tag: str, purpose: str, served_by: str | None) -> float:
    """Append one call to the ledger. Returns its cost."""
    cost = cost_of(usage, model)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"t": time.time(), "tag": tag, "purpose": purpose, "cost": round(cost, 6),
                            "usage": usage, "model": served_by}) + "\n")
    return cost


def text(t: str) -> dict:
    return {"type": "input_text", "text": t}


def image(png: bytes) -> dict:
    """No `detail`, as the base harness sends it (a 256 x 256 frame is 77 tokens at any detail)."""
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode()}


def n_images(content: list[dict]) -> int:
    return sum(x["type"] == "input_image" for x in content)


def output_text(d: dict) -> str:
    out = []
    for item in d.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") in ("output_text", "text"):
                out.append(c.get("text", ""))
    return "".join(out)


def _stored_prefixes() -> dict:
    """The stored prefix responses by key. A file caught mid-write by another process reads as empty."""
    try:
        return json.loads(PREFIX_IDS.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class Client:
    """One model at the Act effort. `ask` makes a call that sees nothing of other calls, `ask_chained` one turn of a
    stored conversation."""

    def __init__(self, model: str, tag: str = "") -> None:
        if model not in PRICES:
            raise CallFailed(f"no price for {model}. Add it to memi/llm/prices.py so the ledger stays right")
        self.model, self.effort, self.tag = model, config.EFFORT, tag
        self.key = api_key()
        self._prefix_lock = threading.Lock()   # one stored prefix per key, not one per worker thread
        self.ctx = ssl.create_default_context(cafile=ca_file())

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {self.key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=self.ctx) as r:
            return json.loads(r.read())

    def _send(self, body: dict, purpose: str) -> dict:
        last = None
        for attempt in range(ATTEMPTS):
            try:
                return self._post(body)
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code not in RETRIED_HTTP:
                    raise CallFailed(last) from e
            except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, json.JSONDecodeError) as e:
                last = repr(e)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise CallFailed(f"{purpose}: {last}")

    def _body(self, system: str, content: list[dict], store: bool) -> dict:
        body = {"model": self.model, "instructions": system, "input": [{"role": "user", "content": content}],
                "store": store}
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        return body

    def _prefix_id(self, system: str, prefix: list[dict], purpose: str, fresh: bool = False) -> str:
        """A stored response holding only `prefix`, so that calls chained to it read it from the cache.

        The provider reads a cached prompt back only for an identical prompt, so calls that share the policy
        prompts and differ in the cycle never hit it. Chained to a stored response that holds the shared part,
        they do, as the base harness's chained turns did. A chained call still sees nothing of other calls.
        """
        key = hashlib.sha256(json.dumps([self.model, self.effort, system, prefix]).encode()).hexdigest()[:24]
        ids = _stored_prefixes()
        if key in ids and not fresh:
            return ids[key]
        content = prefix + [text("Reply with OK. What to answer follows in the next message.")]
        d = self._send(self._body(system, content, store=True), purpose + ".prefix")
        log_call(d.get("usage") or {}, self.model, self.tag, purpose + ".prefix", d.get("model"))
        ids = _stored_prefixes()
        ids[key] = d["id"]
        # Several processes share the file, so it is written whole under another name and renamed into place.
        fd, tmp = tempfile.mkstemp(dir=PREFIX_IDS.parent, prefix=".prefix_ids.")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(ids))
        os.replace(tmp, PREFIX_IDS)
        return d["id"]

    def ask_chained(self, system: str, content: list[dict], purpose: str, previous: str | None) -> dict:
        """One turn of a conversation the provider keeps: stored, and chained to `previous` (the response id of the
        turn before, None for the first). Returns what ask returns, and the response `id` to chain the next turn to."""
        body = self._body(system, content, store=True)
        if previous:
            body["previous_response_id"] = previous
        t0 = time.time()
        d = self._send(body, purpose)
        return {**self._finish(d, purpose, t0), "id": d["id"]}

    def _finish(self, d: dict, purpose: str, t0: float) -> dict:
        """A response's text, parsed answer, usage and cost, its cost written to the ledger."""
        usage = d.get("usage") or {}
        cost = log_call(usage, self.model, self.tag, purpose, d.get("model"))
        t = output_text(d)
        return {"text": t, "json": parse_object(t), "usage": usage, "cost": cost, "model": d.get("model"),
                "s": round(time.time() - t0, 2)}

    def ask(self, system: str, content: list[dict], purpose: str, prefix: list[dict] | None = None) -> dict:
        """One call that sees only `prefix` + `content`. Returns {text, json, usage, cost, model, s}."""
        body = self._body(system, content, store=False)
        t0 = time.time()
        if prefix:
            with self._prefix_lock:
                body["previous_response_id"] = self._prefix_id(system, prefix, purpose)
            try:
                d = self._send(body, purpose)
            except CallFailed as e:                   # an expired stored prefix is stored again, once
                if "previous" not in str(e).lower() and "not found" not in str(e).lower():
                    raise
                with self._prefix_lock:
                    body["previous_response_id"] = self._prefix_id(system, prefix, purpose, fresh=True)
                d = self._send(body, purpose)
        else:
            d = self._send(body, purpose)
        return self._finish(d, purpose, t0)
