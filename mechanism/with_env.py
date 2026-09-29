"""Run a script with extra environment variables: with_env.py K=V [K=V ...] -- SCRIPT ARGS..."""
import os
import runpy
import sys

i = sys.argv.index("--")
for kv in sys.argv[1:i]:
    k, v = kv.split("=", 1)
    os.environ[k] = v
script = sys.argv[i + 1]
sys.argv = [script] + sys.argv[i + 2:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
