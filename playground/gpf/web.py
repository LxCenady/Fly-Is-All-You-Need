"""GPF web UI: a chat-style page with a model picker, served on 127.0.0.1 only.

python -m gpf --web [--port 8765]      then open http://127.0.0.1:8765
Standard library only (http.server); text is streamed to the page as server-sent events.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .models import MODELS, generate

PAGE = (Path(__file__).parent / "web.html").read_text(encoding="utf-8")
_models: dict = {}
_lock = threading.Lock()            # one generation at a time (the connectome model is one GPU state)


def brain_available() -> bool:
    """GPF-1 needs flybrain (GPU build) importable; see gpf/brain.py."""
    from .brain import available
    return available()


def model_list():
    info = {
        "brain": ("GPF-1 (fly connectome)", "166,700 simulated neurons per character, GPU. Pretrained by evolution."),
        "kn7": ("Kneser-Ney 7-gram", "Counts of 7-character sequences in 1M characters. Instant."),
        "kn5-20k": ("Kneser-Ney 5-gram, 20k", "Trained on the same 20k characters as the fly. Fair comparison."),
        "gru": ("GRU", "A small trained recurrent network, 1M characters. The best writer here."),
    }
    out = []
    for key in ("brain", "kn7", "kn5-20k", "gru"):
        name, desc = info[key]
        ok = brain_available() if key == "brain" else True
        out.append({"key": key, "name": name, "desc": desc, "available": ok})
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
                emit({"status": "reading your prompt..."})
                t0 = time.time(); count = [0]

                def on_char(c):
                    count[0] += 1; emit({"c": c})

                generate(model, prompt, n, temp, topk, seed, on_char=on_char, stop=lambda: gone[0])
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
