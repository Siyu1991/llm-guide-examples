#!/usr/bin/env python3
"""GitHub Copilot AI Credits 消耗模拟（离线，不调 API）。

价格与额度取自 docs.github.com（2026-10-30 查阅）:
- 1 AI credit = $0.01
- Pro: $10/月, base 1000 + flex 500 = 1500 credits
- Pro+: $39, 3900+3100=7000
- Max: $100, 10000+10000=20000
- Business: $19/席, 1900 credits/人（池化）
- Enterprise: $39/席, 3900 credits/人
- GPT-6.1 Sol Default 档（≤272K）: in $2 / cache_read $0.10 / cache_write $2.50 / out $10 每百万 token
- Claude Sonnet 5.5: in $2 / cache_read $0.10 / cache_write $2.50 / out $10
- Code completions 不消耗 AI credits（官方说明）

场景是作者构造的用量假设，用来算「额度能撑多少次会话」，不是 GitHub 公布的用户均值。
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

CREDIT_USD = 0.01

PLANS = {
    "Pro": {"price_usd": 10, "credits": 1500},
    "Pro+": {"price_usd": 39, "credits": 7000},
    "Max": {"price_usd": 100, "credits": 20000},
    "Business": {"price_usd": 19, "credits": 1900},
    "Enterprise": {"price_usd": 39, "credits": 3900},
}

# USD per 1M tokens
MODELS = {
    "gpt-6.1-sol": {"in": 2.0, "cache_read": 0.10, "cache_write": 2.50, "out": 10.0},
    "claude-sonnet-5.5": {"in": 2.0, "cache_read": 0.10, "cache_write": 2.50, "out": 10.0},
    "gpt-6-luna": {"in": 0.10, "cache_read": 0.01, "cache_write": 0.125, "out": 0.50},
}


@dataclass
class Turn:
    input_tok: int
    cache_read_tok: int
    cache_write_tok: int
    output_tok: int


def usd_for_turn(model: str, t: Turn) -> float:
    p = MODELS[model]
    return (
        t.input_tok * p["in"]
        + t.cache_read_tok * p["cache_read"]
        + t.cache_write_tok * p["cache_write"]
        + t.output_tok * p["out"]
    ) / 1_000_000


def credits_for_usd(usd: float) -> float:
    return usd / CREDIT_USD


def session_quick_chat(model: str, turns: int = 3) -> dict:
    """短问答：每轮新输入 800，输出 300，无缓存。"""
    turns_data = [Turn(800, 0, 0, 300) for _ in range(turns)]
    usd = sum(usd_for_turn(model, t) for t in turns_data)
    return {"name": "quick_chat", "turns": turns, "usd": usd, "credits": credits_for_usd(usd)}


def session_agent_long(model: str, turns: int = 12) -> dict:
    """长 Agent：稳定前缀 6000 首轮全写，其后读累积前缀 + 每轮新增 1500，输出 600。"""
    prefix = 6000
    new_per = 1500
    out_per = 600
    usd = 0.0
    for i in range(turns):
        if i == 0:
            t = Turn(0, 0, prefix + new_per, out_per)  # 简化：首轮整段当 cache_write+普通；按 write 计
            # 更贴官方：无缓存时全按 input；这里用「首轮 input，其后 cache」
            t = Turn(prefix + new_per, 0, 0, out_per)
        else:
            cached = prefix + new_per * i
            t = Turn(new_per, cached, 0, out_per)
        usd += usd_for_turn(model, t)
    return {"name": "agent_long", "turns": turns, "usd": usd, "credits": credits_for_usd(usd)}


def sessions_until_empty(plan: str, session: dict) -> dict:
    c = PLANS[plan]["credits"]
    n = int(c // session["credits"]) if session["credits"] > 0 else 0
    rem = c - n * session["credits"]
    return {
        "plan": plan,
        "plan_credits": c,
        "session": session["name"],
        "session_credits": round(session["credits"], 4),
        "full_sessions": n,
        "remainder_credits": round(rem, 4),
    }


def main():
    results = {"plans": PLANS, "models": MODELS, "credit_usd": CREDIT_USD, "scenarios": []}
    for model in ("gpt-6.1-sol", "gpt-6-luna"):
        quick = session_quick_chat(model)
        agent = session_agent_long(model)
        block = {"model": model, "quick_chat": quick, "agent_long": agent, "capacity": []}
        for plan in PLANS:
            block["capacity"].append(sessions_until_empty(plan, quick))
            block["capacity"].append(sessions_until_empty(plan, agent))
        results["scenarios"].append(block)

    # 组织侧：50 人 Business，池化 50*1900；对比「每人隔离」浪费
    org_n = 50
    pooled = org_n * PLANS["Business"]["credits"]
    # 假设用量重尾：10 人用掉人均 3 倍，40 人只用人均 0.4 倍（相对 agent_long on sol）
    agent = session_agent_long("gpt-6.1-sol")
    heavy = 10 * 3 * agent["credits"]
    light = 40 * 0.4 * agent["credits"]
    isolated_waste = max(0, 40 * (PLANS["Business"]["credits"] - 0.4 * agent["credits"]))
    # 隔离时轻用户剩余额度不能给重用户；池化可以
    results["org_pool"] = {
        "seats": org_n,
        "pooled_credits": pooled,
        "heavy_plus_light_credits": round(heavy + light, 2),
        "fits_in_pool": (heavy + light) <= pooled,
        "note": "重尾假设为教学构造，非 GitHub 公布分布",
    }

    path = os.path.join(OUT, "ep18_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results["scenarios"][0]["quick_chat"], indent=2))
    print(json.dumps(results["scenarios"][0]["agent_long"], indent=2))
    print("org_pool", results["org_pool"])
    print("wrote", path)


if __name__ == "__main__":
    main()
