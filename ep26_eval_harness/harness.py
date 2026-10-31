#!/usr/bin/env python3
"""离线回归评测：固定黄金集 + 多「模型」版本对比。不需要 API key。"""
from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# 黄金集：问答 + 可接受答案集合（小写规范化后 exact / 包含匹配）
GOLDEN = [
    {"id": "g1", "q": "退货窗口几天？", "accept": ["7天", "七天"], "metric": "contains"},
    {"id": "g2", "q": "订单号格式？", "accept": ["ord-"], "metric": "contains"},
    {"id": "g3", "q": "是否支持货到付款？", "accept": ["不支持", "不支持货到付款"], "metric": "contains"},
    {"id": "g4", "q": "客服工作时间？", "accept": ["9:00-18:00", "9点到18点"], "metric": "contains"},
    {"id": "g5", "q": "默认运费多少？", "accept": ["12元", "12 元"], "metric": "contains"},
]

# 模拟两个模型版本的输出（v2 故意在 g3/g5 退化）
MOCK_MODELS = {
    "model_v1": {
        "g1": "退货窗口是7天，签收次日起算。",
        "g2": "订单号形如 ORD-123456。",
        "g3": "目前不支持货到付款。",
        "g4": "客服工作时间 9:00-18:00（工作日）。",
        "g5": "默认运费 12元，满99包邮。",
    },
    "model_v2": {
        "g1": "退货窗口是7天。",
        "g2": "请使用 ORD- 前缀的订单号。",
        "g3": "部分地区支持货到付款。",  # 相对黄金集错误
        "g4": "工作日 9:00-18:00 在线。",
        "g5": "运费以结算页为准。",  # 未命中固定数字
    },
}

# 粗糙的 LLM-as-judge mock：偏好更长、含「详」的回答（演示偏见）
def mock_judge_prefer_verbose(answer: str, gold_hints: list[str]) -> float:
    score = 0.3
    if any(h.lower() in answer.lower() for h in gold_hints):
        score += 0.4
    score += min(len(answer) / 80.0, 0.3)  # 偏爱长文
    return round(min(score, 1.0), 3)


def score_case(answer: str, case: dict) -> bool:
    a = answer.lower()
    return any(x.lower() in a for x in case["accept"])


@dataclass
class CaseResult:
    case_id: str
    pass_exactish: bool
    judge_score: float
    answer: str


def eval_model(name: str, outputs: dict) -> dict:
    results = []
    for case in GOLDEN:
        ans = outputs[case["id"]]
        results.append(CaseResult(
            case_id=case["id"],
            pass_exactish=score_case(ans, case),
            judge_score=mock_judge_prefer_verbose(ans, case["accept"]),
            answer=ans,
        ))
    n = len(results)
    pass_rate = sum(r.pass_exactish for r in results) / n
    judge_mean = sum(r.judge_score for r in results) / n
    return {
        "model": name,
        "pass_rate": round(pass_rate, 4),
        "judge_mean": round(judge_mean, 4),
        "cases": [asdict(r) for r in results],
    }


def regression(base: dict, cand: dict, pass_drop_tol: float = 0.0) -> dict:
    drop = base["pass_rate"] - cand["pass_rate"]
    failed = drop > pass_drop_tol
    return {
        "base": base["model"],
        "candidate": cand["model"],
        "pass_rate_base": base["pass_rate"],
        "pass_rate_candidate": cand["pass_rate"],
        "pass_rate_drop": round(drop, 4),
        "gate_failed": failed,
        "note": "以黄金集 contains 命中率为门禁；judge_mean 仅作对照，不单独放行",
    }


def main():
    reports = {k: eval_model(k, v) for k, v in MOCK_MODELS.items()}
    gate = regression(reports["model_v1"], reports["model_v2"])
    out = {"golden_size": len(GOLDEN), "reports": reports, "regression_gate": gate}
    path = os.path.join(OUT, "ep26_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=== 回归评测 ===")
    for name, r in reports.items():
        print(f"{name}: pass_rate={r['pass_rate']:.0%} judge_mean={r['judge_mean']:.3f}")
        for c in r["cases"]:
            mark = "PASS" if c["pass_exactish"] else "FAIL"
            print(f"  {c['case_id']} {mark} judge={c['judge_score']}")
    print(f"门禁: drop={gate['pass_rate_drop']:.0%} gate_failed={gate['gate_failed']}")
    print("写入", path)


if __name__ == "__main__":
    main()
