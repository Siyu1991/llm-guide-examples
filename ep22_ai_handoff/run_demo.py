#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
out_csv = OUT / "summary.csv"
if out_csv.exists():
    out_csv.unlink()
rc = subprocess.call([sys.executable, str(HERE / "csv_summary.py"), str(HERE / "testdata/ok.csv"), str(out_csv)])
test = subprocess.run([sys.executable, str(HERE / "test_csv_summary.py")], capture_output=True, text=True)
result = {
    "summarize_rc": rc,
    "output_csv": out_csv.read_text(encoding="utf-8") if out_csv.exists() else None,
    "unittest_rc": test.returncode,
    "unittest_stdout": test.stderr + test.stdout,
}
(OUT / "ep22_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
