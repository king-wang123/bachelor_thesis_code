#!/usr/bin/env bash
set -euo pipefail

ENV=/data/250010072/zlh/conda_envs/verl
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=/data/250010072/zlh/king/verl

echo "[tools]"
which python3 || true
which gcc || true
which g++ || true
which javac || true
which java || true
which go || true
go version || true
javac -version || true
java -version || true

echo "[python]"
"$ENV/bin/python" - <<'PY'
import sys
mods = ["verl", "vllm", "torch", "ray", "pandas", "pyarrow", "datasets", "requests"]
print("python", sys.executable)
for m in mods:
    try:
        mod = __import__(m)
        print(m, getattr(mod, "__version__", "ok"), getattr(mod, "__file__", ""))
    except Exception as e:
        print(m, "ERR", repr(e))
PY
