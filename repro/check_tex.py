"""Static checks of the papers' LaTeX (no TeX installation needed).

    python repro/check_tex.py paper/main.tex paper_lm/main.tex

Reports unbalanced environments, citations without a bibitem, unused bibitems, references to
missing labels, unbalanced braces, and leftover placeholders.  Exit code 1 on any problem.
"""
import re
import sys

ENVS = ("itemize", "enumerate", "table", "table*", "figure", "figure*", "tabular", "abstract")
PLACEHOLDER = re.compile(r"\b(VH\d|FLY[A-Z]+|TODO|XXX|REPRO)\b")


def check(path):
    s = open(path, encoding="utf-8").read()
    body = re.sub(r"(?<!\\)%.*", "", s)                     # drop comments
    problems = []
    for e in ENVS:
        b, n = body.count("\\begin{" + e + "}"), body.count("\\end{" + e + "}")
        if b != n:
            problems.append(f"environment {e}: {b} begin vs {n} end")
    cited = {c.strip() for x in re.findall(r"\\cite\{([^}]+)\}", body) for c in x.split(",")}
    keys = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for name, missing in (("citation without bibitem", cited - keys), ("unused bibitem", keys - cited),
                          ("reference to missing label", refs - labels)):
        problems += [f"{name}: {m}" for m in sorted(missing)]
    depth = body.count("{") - body.count("\\{") - (body.count("}") - body.count("\\}"))
    if depth:
        problems.append(f"brace balance {depth:+d}")
    text = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("\\newcommand"))
    problems += [f"placeholder left: {m}" for m in sorted(set(PLACEHOLDER.findall(text)))]
    return problems


def main(paths):
    bad = 0
    for p in paths:
        probs = check(p)
        print(f"{p}: {'ok' if not probs else ''}")
        for x in probs:
            print("   ", x)
        bad += len(probs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
