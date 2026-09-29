"""GPF desktop entry point (used for the packaged builds).

Double-click / run without arguments: start the web UI on a free local port and open it in the
browser.  gpf --cli ...  : command-line generation (see gpf/__main__.py).
gpf train ...  : train your own model (see gpf/train.py; the Lite build trains n-grams only).
"""
import socket
import sys
import threading
import webbrowser


def free_port(preferred=8765):
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def main():
    if sys.argv[1:2] == ["train"]:
        from gpf.train import main as train_main
        train_main(sys.argv[2:])
        return
    if "--cli" in sys.argv:
        from gpf.__main__ import cli
        cli(sys.argv[1:])
        return
    from gpf.web import main as web_main
    port = free_port()
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    web_main(port)


if __name__ == "__main__":
    main()
