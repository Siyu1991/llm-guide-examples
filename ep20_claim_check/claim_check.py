#!/usr/bin/env python3
"""主张拆解与来源依赖检查（离线）。

输入一份虚构报告片段 + 引用元数据，输出：
1) 比较级主张拆成可核查子句
2) 多条 URL 追到同一 root_id 时，独立证据数 = 1
3) 按「每请求」与「每合格结果」重算单价（算术演示）

数字与网页关系均为教学构造，脚本不访问网络。
"""
from __future__ import annotations
import json, os, re
from dataclasses import dataclass, asdict

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


@dataclass
class Source:
    url: str
    root_id: str  # 同一原始实验/公告共享同一 root
    published: str
    supports: list[str]  # 能支持的原子主张 id


@dataclass
class AtomClaim:
    id: str
    text: str
    kind: str  # fact | comparison | recommendation


def split_comparison(sentence: str) -> list[str]:
    """极简拆解：按顿号/逗号切开比较维度。"""
    s = sentence.strip()
    parts = re.split(r"[，,、]|也|并且|而且", s)
    return [p.strip() for p in parts if p.strip()]


def independent_evidence_count(sources: list[Source], claim_id: str) -> dict:
    supporting = [s for s in sources if claim_id in s.supports]
    roots = {s.root_id for s in supporting}
    return {
        "claim_id": claim_id,
        "link_count": len(supporting),
        "independent_roots": len(roots),
        "root_ids": sorted(roots),
        "urls": [s.url for s in supporting],
    }


def cost_per_success(price_per_req: float, success_rate: float) -> float:
    if success_rate <= 0:
        raise ValueError("success_rate must be > 0")
    return price_per_req / success_rate


def demo() -> dict:
    atoms = [
        AtomClaim("c1", "工具A提供自托管选项", "fact"),
        AtomClaim("c2", "工具A比工具B更便宜", "comparison"),
        AtomClaim("c3", "中小团队应采用工具A", "recommendation"),
    ]
    sources = [
        Source("https://vendor.example/a/features", "vendor-a-features", "2026-01-10", ["c1"]),
        Source("https://news.example/repost-1", "vendor-a-bench-90", "2026-03-01", ["c2"]),
        Source("https://news.example/repost-2", "vendor-a-bench-90", "2026-03-02", ["c2"]),
        Source("https://vendor.example/blog/internal-90", "vendor-a-bench-90", "2026-02-28", ["c2"]),
    ]
    sentence = "这个方案更快、更便宜，也更准确"
    split = split_comparison(sentence)
    ev = independent_evidence_count(sources, "c2")
    # 单价演示：A 0.005 元/请求 成功率 0.5；B 0.01 成功率 0.9
    a_req, a_p = 0.005, 0.5
    b_req, b_p = 0.010, 0.9
    costs = {
        "A_per_req": a_req,
        "B_per_req": b_req,
        "A_per_success": cost_per_success(a_req, a_p),
        "B_per_success": cost_per_success(b_req, b_p),
        "ratio_per_req": a_req / b_req,
        "ratio_per_success": cost_per_success(a_req, a_p) / cost_per_success(b_req, b_p),
    }
    out = {
        "atoms": [asdict(a) for a in atoms],
        "split_comparison": split,
        "evidence_c2": ev,
        "cost_demo": costs,
        "note": "sources 与价格均为虚构教学数据",
    }
    with open(os.path.join(OUT, "ep20_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
