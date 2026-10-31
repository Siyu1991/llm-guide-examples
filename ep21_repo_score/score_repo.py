#!/usr/bin/env python3
"""对本地 fixture 仓库做采纳前检查打分（离线）。

检查项：README 是否写明 CPU/RAM/Docker、是否有许可证句、CI 矩阵是否含多 OS、
最近运行是否大量 skip。分数为启发式教学分，不是安全评级。
"""
from __future__ import annotations
import json, os, re

HERE = os.path.dirname(__file__)
FIX = os.path.join(HERE, "fixture")
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def score_readme(text: str) -> dict:
    checks = {
        "mentions_cpu": bool(re.search(r"CPU\s*>=?\s*\d", text, re.I)),
        "mentions_ram": bool(re.search(r"RAM\s*>=?\s*\d", text, re.I)),
        "mentions_docker": "docker" in text.lower(),
        "has_quick_start": "quick start" in text.lower() or "快速" in text,
        "mentions_license": "license" in text.lower() or "许可" in text,
    }
    score = sum(10 for v in checks.values() if v)
    return {"checks": checks, "score": score, "max": 50}


def score_ci(data: dict) -> dict:
    os_list = data.get("os") or []
    runs = data.get("recent_runs") or []
    multi_os = len(os_list) >= 2
    skip_n = sum(1 for r in runs if r.get("tests_skipped"))
    success_n = sum(1 for r in runs if r.get("conclusion") == "success")
    checks = {
        "multi_os_matrix": multi_os,
        "os_list": os_list,
        "recent_success": success_n,
        "recent_total": len(runs),
        "skipped_runs": skip_n,
        "skip_ratio": (skip_n / len(runs)) if runs else 0.0,
    }
    score = 0
    if multi_os:
        score += 20
    else:
        score += 5  # 单 OS 有矩阵仍给一点分
    if runs and skip_n / len(runs) <= 0.5:
        score += 20
    elif runs:
        score += 8
    return {"checks": checks, "score": score, "max": 40}


def main():
    readme = open(os.path.join(FIX, "README.md"), encoding="utf-8").read()
    ci = json.load(open(os.path.join(FIX, "ci_matrix.json"), encoding="utf-8"))
    r = score_readme(readme)
    c = score_ci(ci)
    total = r["score"] + c["score"]
    out = {
        "readme": r,
        "ci": c,
        "total": total,
        "total_max": r["max"] + c["max"],
        "verdict": "可以进入小规模试点" if total >= 50 else "先补齐安装条件或 CI 覆盖再试点",
    }
    path = os.path.join(OUT, "ep21_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
