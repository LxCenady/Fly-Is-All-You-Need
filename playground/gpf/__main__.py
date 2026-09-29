"""python -m gpf            -> terminal UI
python -m gpf --cli --model brain --prompt "ROMEO:\\n" --n 300 --temp 0.7"""
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
    model = MODELS[a.model][1](print)
    prompt = a.prompt.replace("\\n", "\n")
    print(prompt, end="", flush=True)
    generate(model, prompt, a.n, a.temp, a.topk, a.seed, on_char=lambda c: print(c, end="", flush=True))
    print()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli(sys.argv[1:])
    else:
        from .tui import main
        main()
