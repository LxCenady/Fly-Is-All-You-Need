"""python -m uctf import ...   build a connectome folder from CSV files (importer.py)
python -m uctf bench ...    the standard comparison on a text (bench.py)"""
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["import"]:
        from .importer import main as run
    elif argv[:1] == ["bench"]:
        from .bench import main as run
    else:
        print(__doc__)
        return
    run(argv[1:])


if __name__ == "__main__":
    main()
