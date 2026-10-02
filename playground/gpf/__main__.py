"""python -m gpf            -> terminal UI
python -m gpf --web      -> web UI
python -m gpf --cli --model brain --prompt "ROMEO:\\n" --n 300 --temp 0.7
python -m gpf train kn|gru|brain --data my.txt --name my-model   (see gpf/train.py)
python -m gpf bench --substrate malecns-v1 --data my.txt           (UCTF: uctf/uctf/bench.py)
python -m gpf import --neurons n.csv --edges e.csv --out DIR       (UCTF: uctf/uctf/importer.py)
python -m gpf get-brain                    download the GPF-1 connectome for the CPU (~138 MB, no GPU needed)
python -m gpf get-brain --from-flybrain DIR   build it from flybrain's data instead (needs scipy)"""
import argparse
import sys

from .models import MODELS, generate


def cli(argv):
    ap = argparse.ArgumentParser(prog="gpf")
    ap.add_argument("--cli", action="store_true")
    ap.add_argument("--model", choices=list(MODELS), default="kn7")
    ap.add_argument("--prompt", default="ROMEO:\\n")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):        # any script, also when piped on Windows
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    model = MODELS[a.model][1](print)
    prompt = a.prompt.replace("\\n", "\n")
    print(prompt, end="", flush=True)
    generate(model, prompt, a.n, a.temp, a.topk, a.seed, on_char=lambda c: print(c, end="", flush=True))
    print()


def get_brain(argv):
    """GPF-1 on the CPU needs the connectome bundle (gpf/cpu_brain.py)."""
    from pathlib import Path
    from .cpu_brain import BUNDLE, fetch, prepare
    if "--from-flybrain" in argv:
        dest = Path.home() / ".gpf" / BUNDLE
        dest.parent.mkdir(parents=True, exist_ok=True)
        prepare(argv[argv.index("--from-flybrain") + 1], dest)
        print(f"wrote {dest}")
    else:
        fetch()


def run(argv):
    if argv[:1] == ["train"]:
        from .train import main as train_main
        train_main(argv[1:])
    elif argv[:1] == ["bench"]:
        from uctf.bench import main as bench_main
        bench_main(argv[1:])
    elif argv[:1] == ["import"]:
        from uctf.importer import main as import_main
        import_main(argv[1:])
    elif argv[:1] == ["get-brain"]:
        get_brain(argv[1:])
    elif "--web" in argv:
        from .web import main as web_main
        port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8765
        web_main(port)
    elif "--cli" in argv:
        cli(argv)
    else:
        from .tui import main
        main()


if __name__ == "__main__":
    run(sys.argv[1:])
