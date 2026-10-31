#!/usr/bin/env python3
"""CSV 按月汇总：Decimal 金额、严格日期、默认拒绝覆盖。

验收条件写在 test_csv_summary.py。只用标准库。
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from datetime import datetime
from pathlib import Path


def summarize(in_path: Path, out_path: Path, force: bool = False) -> int:
    if out_path.exists() and not force:
        print(f"error: output exists: {out_path} (use --force)", file=sys.stderr)
        return 2
    totals: dict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    with in_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "date" not in reader.fieldnames or "amount" not in reader.fieldnames:
            print("error: need date,amount columns", file=sys.stderr)
            return 1
        for i, row in enumerate(reader, start=2):
            raw_date = (row.get("date") or "").strip()
            raw_amt = (row.get("amount") or "").strip()
            if not raw_date and not raw_amt:
                continue
            try:
                dt = datetime.strptime(raw_date, "%Y-%m-%d")
            except ValueError:
                print(f"error: line {i}: bad date {raw_date!r}", file=sys.stderr)
                return 1
            try:
                amt = Decimal(raw_amt)
            except InvalidOperation:
                print(f"error: line {i}: bad amount {raw_amt!r}", file=sys.stderr)
                return 1
            key = dt.strftime("%Y-%m")
            totals[key] += amt
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "amount"])
        for k in sorted(totals):
            w.writerow([k, str(totals[k])])
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    return summarize(Path(args.input), Path(args.output), force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
