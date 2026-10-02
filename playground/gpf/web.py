"""GPF web UI: a chat-style page with a model picker, served on 127.0.0.1 only.

python -m gpf --web [--port 8765]      then open http://127.0.0.1:8765
Standard library only (http.server); text is streamed to the page as server-sent events.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .models import MODELS, generate, refresh_models

PAGE = (Path(__file__).parent / "web.html").read_text(encoding="utf-8")
LOGO = (Path(__file__).parent / "logo.svg")
MAX_DOTS = 8000                     # most neurons drawn per character in the brain view
_rng = np.random.default_rng(0)
_cache: dict = {}
_models: dict = {}
_lock = threading.Lock()            # one generation at a time (the connectome model is one simulation state)


def brain_available(key: str = "brain") -> bool:
    """GPF-1 runs on the GPU (flybrain + CuPy) or on the CPU from the bundled connectome
    (gpf/cpu_brain.py).  A user's connectome model is built by UCTF, which also needs scipy."""
    from .brain import available
    if key == "brain":
        return available()
    import importlib.util
    return importlib.util.find_spec("scipy") is not None


def model_list():
    """Built-in models first, then the user's own (python -m gpf train ...); re-scanned on every
    call so a model trained while the server runs shows up after a page reload."""
    refresh_models()
    out = []
    for key, (label, _, desc, needs_brain) in MODELS.items():
        brain_ok = brain_available(key) if needs_brain else True
        name = {"brain": "GPF-1 (fly connectome)", "kn7": "Kneser-Ney 7-gram",
                "kn5-20k": "Kneser-Ney 5-gram, 20k", "gru": "GRU"}.get(key, label)
        out.append({"key": key, "name": name, "desc": desc, "available": bool(brain_ok),
                    "user": key.startswith("user:"), "brain": needs_brain})
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                  # keep the console quiet
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._send(200, json.dumps(model_list()), "application/json")
        elif self.path == "/logo.svg" and LOGO.exists():
            self._send(200, LOGO.read_bytes(), "image/svg+xml")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/generate":
            self._send(404, "not found", "text/plain"); return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            key = req.get("model", "kn7")
            if key not in MODELS:
                raise ValueError("unknown model")
            prompt = str(req.get("prompt", ""))[:4000]
            n = max(1, min(int(req.get("n", 300)), 2000))
            temp = max(0.05, min(float(req.get("temp", 0.7)), 2.0))
            topk = max(0, min(int(req.get("topk", 0)), 65))
            seed = int(req.get("seed", 0))
            brainview = bool(req.get("brainview", False))
            need_map = bool(req.get("need_map", False))
        except (ValueError, TypeError) as e:
            self._send(400, json.dumps({"error": str(e)}), "application/json"); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        gone = [False]

        def emit(obj):
            if gone[0]:
                return
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8")); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                gone[0] = True                  # the page pressed Stop or closed

        with _lock:
            try:
                if key not in _models:
                    emit({"status": "loading " + key + "..."})
                    _models[key] = MODELS[key][1](lambda m: emit({"status": m}))
                model = _models[key]
                watch = brainview and hasattr(model, "map_payload")
                on_feed = None
                if hasattr(model, "record_activity"):
                    model.record_activity(watch)
                if watch:
                    if need_map:
                        if ("map", key) not in _cache:              # one map per connectome model
                            _cache[("map", key)] = model.map_payload()
                        emit({"brainmap": _cache[("map", key)]})

                    def on_feed(c, phase):
                        a = model.activity()
                        if a is None:
                            return
                        ids = a["ids"]
                        if len(ids) > MAX_DOTS:                       # cap what is drawn, like the fly.ai view
                            ids = np.sort(_rng.choice(ids, MAX_DOTS, replace=False))
                        emit({"act": base64.b64encode(ids.astype(np.uint32).tobytes()).decode(),
                              "ch": c, "phase": phase, "spikes": a["spikes"], "active": a["active"]})
                emit({"status": "reading your prompt..."})
                t0 = time.time(); count = [0]

                def on_char(c):
                    count[0] += 1; emit({"c": c})

                generate(model, prompt, n, temp, topk, seed, on_char=on_char, stop=lambda: gone[0],
                         on_feed=on_feed)
                dt = time.time() - t0
                emit({"done": True, "chars": count[0], "seconds": round(dt, 2),
                      "rate": round((count[0] + len(prompt)) / max(dt, 1e-9), 1), "model": model.name})
            except Exception as e:              # noqa: BLE001
                emit({"error": str(e)})


def main(port=8765):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"GPF web UI on http://127.0.0.1:{port}  (ctrl+c to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
