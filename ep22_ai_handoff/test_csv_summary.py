#!/usr/bin/env python3
import os, tempfile, unittest, csv
from pathlib import Path
from csv_summary import summarize

HERE = Path(__file__).parent
DATA = HERE / "testdata"

class TestCSV(unittest.TestCase):
    def test_ok_sum(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.csv"
            rc = summarize(DATA / "ok.csv", out)
            self.assertEqual(rc, 0)
            rows = list(csv.DictReader(out.open()))
            by = {r["month"]: r["amount"] for r in rows}
            self.assertEqual(by["2026-01"], "13.10")
            self.assertEqual(by["2026-02"], "48.35")

    def test_bad_date(self):
        with tempfile.TemporaryDirectory() as d:
            rc = summarize(DATA / "bad_date.csv", Path(d) / "out.csv")
            self.assertEqual(rc, 1)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out.csv"
            out.write_text("x", encoding="utf-8")
            rc = summarize(DATA / "ok.csv", out, force=False)
            self.assertEqual(rc, 2)
            rc2 = summarize(DATA / "ok.csv", out, force=True)
            self.assertEqual(rc2, 0)

if __name__ == "__main__":
    unittest.main()
