"""GPF desktop entry point (used for the packaged builds).

    gpf                    start the web UI on a free local port and open it in the browser
    gpf --no-browser       the same, without opening the browser
    gpf --cli ...          command-line generation (gpf --cli --help)
    gpf train ...          train your own model (the Lite build trains n-grams only)
    gpf bench | import     UCTF's benchmark and importer (full install; they need scipy)

Anything else is an error: the web UI starts only when asked for.
"""
import importlib.util
import socket
import sys
import threading
import webbrowser

WEB_FLAGS = {"--no-browser", "--web"}


def free_port(preferred=8765):
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    return preferred


def uctf_command(argv):
    """bench / import: UCTF needs scipy, which the Lite build leaves out."""
    if importlib.util.find_spec("scipy") is None:
        sys.exit(f"gpf {argv[0]} needs the full install (Python with numpy and scipy); "
                 "the Lite build trains and runs n-gram models only.\n"
                 "See https://github.com/LxCenady/Fly-Is-All-You-Need/tree/main/playground")
    from gpf.__main__ import run
    run(argv)


def web(argv):
    from gpf.web import main as web_main
    port = free_port()
    if "--no-browser" not in argv:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    web_main(port)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["train"]:
        from gpf.train import main as train_main
        train_main(argv[1:])
    elif argv[:1] in (["bench"], ["import"]):
        uctf_command(argv)
    elif "--cli" in argv:
        from gpf.__main__ import cli
        cli(argv)
    elif set(argv) <= WEB_FLAGS:
        web(argv)
    elif argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
    else:
        sys.exit(f"gpf: unknown arguments {' '.join(argv)}\n\n{__doc__}")


if __name__ == "__main__":
    main()
