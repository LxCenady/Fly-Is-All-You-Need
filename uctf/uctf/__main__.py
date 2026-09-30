"""python -m uctf import ...   CSV files -> a connectome folder (importer.py)
python -m uctf bench ...    the standard comparison on a text (bench.py)
python -m uctf plugins      list the registered plugins of every kind
python -m uctf check SPEC   load a spec, build its network, run one token"""
import sys


def plugins(argv):
    from .core.registry import available, skipped
    for kind, names in available().items():
        print(f"{kind:10} {', '.join(names) or '-'}")
    for module, reason in skipped().items():
        print(f"not loaded: {module} ({reason})")


def check(argv):
    """Load a spec and build its network: the quickest test of a new
    connectome and spec."""
    from .core.network import Network
    if not argv:
        sys.exit("usage: python -m uctf check SPEC [--device cpu]")
    device = argv[argv.index("--device") + 1] if "--device" in argv else "auto"
    net = Network(argv[0], n_tokens=2, device=device)
    x = net.step_token(0)
    layers = ", ".join(f"{k} ({v.matrix.nnz:,})" for k, v in net.cx.layers.items())
    print(f"{net.cx.name}: {net.cx.n:,} neurons; layers: {layers}")
    print(f"readout: {net.readout_sizes} -> {x.size} features per token")


COMMANDS = {"import": "importer", "bench": "bench"}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    if cmd in COMMANDS:
        from importlib import import_module
        import_module(f"uctf.{COMMANDS[cmd]}").main(rest)
    elif cmd == "plugins":
        plugins(rest)
    elif cmd == "check":
        check(rest)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
