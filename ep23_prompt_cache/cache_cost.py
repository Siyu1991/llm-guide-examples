#!/usr/bin/env python3
"""提示缓存成本模拟：多轮 Agent 会话，对比有无缓存的输入费用。

价格取自各厂商官方文档（2026-10-30 查阅，美元 / 百万 token）：
  OpenAI gpt-6.1-sol Standard 短上下文：
    输入 2.00，缓存读 0.10，30 分钟写 2.50，输出 10.00
    https://developers.openai.com/api/docs/pricing
  Anthropic Claude Sonnet 5.5：
    输入 2.00，5 分钟写 2.50，缓存读 0.10，输出 10.00
    https://platform.claude.com/docs/en/about-claude/pricing
  Google Gemini 3.8 Flash（付费档，至 2026-12-31）：
    输入 0.75，缓存读 0.075，输出 3.75
    另有存储费 0.50 / 百万 token / 小时（本模拟会话内忽略）
    https://ai.google.dev/gemini-api/docs/pricing
  DeepSeek deepseek-flash 空闲时段：
    缓存命中 0.003，缓存未命中 0.15，输出 0.60
    https://api-docs.deepseek.com/quick_start/pricing

场景为虚构：稳定系统提示 + 工具定义 8,000 token，每轮新增用户/工具结果
约 1,200 token，输出约 400 token，共 12 轮。缓存前缀按「精确匹配」累积。
不调用任何 API，不需要密钥。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# 美元 / 百万 token
PRICES = {
    "openai_gpt61_sol": {
        "label": "OpenAI gpt-6.1-sol",
        "input": 2.00,
        "cache_read": 0.10,
        "cache_write": 2.50,  # 30m TTL
        "output": 10.00,
        "has_write_premium": True,
    },
    "anthropic_sonnet55": {
        "label": "Anthropic Claude Sonnet 5.5",
        "input": 2.00,
        "cache_read": 0.10,
        "cache_write": 2.50,  # 5m TTL
        "output": 10.00,
        "has_write_premium": True,
    },
    "gemini_38_flash": {
        "label": "Google Gemini 3.8 Flash",
        "input": 0.75,
        "cache_read": 0.075,
        "cache_write": 0.75,  # 首次写入按普通输入计；另有存储费，本模拟忽略
        "output": 3.75,
        "has_write_premium": False,
    },
    "deepseek_flash_offpeak": {
        "label": "DeepSeek flash（空闲）",
        "input": 0.15,  # cache miss
        "cache_read": 0.003,
        "cache_write": 0.15,  # miss = write equivalent
        "output": 0.60,
        "has_write_premium": False,
    },
}

STABLE_PREFIX = 8_000  # 系统提示 + 工具定义
NEW_PER_TURN = 1_200   # 每轮新增输入
OUTPUT_PER_TURN = 400
TURNS = 12


@dataclass
class TurnBill:
    turn: int
    input_tokens: int
    cached_tokens: int
    write_tokens: int
    ordinary_tokens: int
    output_tokens: int
    cost_with_cache: float
    cost_without_cache: float


def usd(tokens: float, price_per_mtok: float) -> float:
    return tokens * price_per_mtok / 1_000_000


def simulate(provider: str) -> dict:
    p = PRICES[provider]
    turns: list[TurnBill] = []
    # 已写入缓存、可被后续命中的前缀长度
    cached_len = 0
    total_with = 0.0
    total_without = 0.0
    total_cached = 0
    total_write = 0
    total_ordinary = 0
    total_out = 0
    total_in = 0

    for t in range(1, TURNS + 1):
        # 第 t 轮时，上下文 = 稳定前缀 + (t-1) 轮历史增量 + 本轮新增
        history = (t - 1) * NEW_PER_TURN
        inp = STABLE_PREFIX + history + NEW_PER_TURN
        # 可命中的部分：稳定前缀 + 已缓存的历史（假设每轮结束后前缀被写入）
        # 首轮：写入整个 inp；之后：读取已缓存前缀，写入本轮新增
        if t == 1:
            read = 0
            write = inp if p["has_write_premium"] or True else 0
            # 对 OpenAI/Anthropic：首轮全部按 cache write；对 Gemini/DeepSeek：首轮按 miss/input
            if p["has_write_premium"]:
                write = inp
                ordinary = 0
            else:
                write = 0
                ordinary = inp
            cached_len = inp
        else:
            # 前缀精确匹配：可读取到上一轮结束时的全部前缀
            read = cached_len
            new_part = inp - cached_len
            if p["has_write_premium"]:
                write = new_part
                ordinary = 0
            else:
                write = 0
                ordinary = new_part
            cached_len = inp

        out = OUTPUT_PER_TURN
        if p["has_write_premium"]:
            c_in = usd(read, p["cache_read"]) + usd(write, p["cache_write"]) + usd(ordinary, p["input"])
        else:
            c_in = usd(read, p["cache_read"]) + usd(ordinary, p["input"])
        c_out = usd(out, p["output"])
        with_c = c_in + c_out
        without = usd(inp, p["input"]) + c_out

        turns.append(TurnBill(
            turn=t, input_tokens=inp, cached_tokens=read, write_tokens=write,
            ordinary_tokens=ordinary if not p["has_write_premium"] else 0,
            output_tokens=out, cost_with_cache=with_c, cost_without_cache=without,
        ))
        total_with += with_c
        total_without += without
        total_cached += read
        total_write += write
        total_ordinary += (ordinary if not p["has_write_premium"] else 0)
        total_out += out
        total_in += inp

    save = total_without - total_with
    return {
        "provider": provider,
        "label": p["label"],
        "prices": {k: v for k, v in p.items() if k != "label"},
        "scenario": {
            "stable_prefix": STABLE_PREFIX,
            "new_per_turn": NEW_PER_TURN,
            "output_per_turn": OUTPUT_PER_TURN,
            "turns": TURNS,
        },
        "summary": {
            "total_input_tokens_billed_as_context": total_in,
            "total_cached_tokens": total_cached,
            "total_write_tokens": total_write,
            "total_ordinary_tokens": total_ordinary,
            "total_output_tokens": total_out,
            "cost_with_cache_usd": round(total_with, 6),
            "cost_without_cache_usd": round(total_without, 6),
            "savings_usd": round(save, 6),
            "savings_pct": round(100 * save / total_without, 2) if total_without else 0,
        },
        "turns": [asdict(x) for x in turns],
    }


def parse_usage_example() -> dict:
    """演示如何从官方响应字段估算费用（虚构 usage，价格用 gpt-6.1-sol）。"""
    # 来自 OpenAI Prompt Caching 文档的字段结构示意
    usage = {
        "input_tokens": 12000,
        "output_tokens": 400,
        "input_tokens_details": {
            "cached_tokens": 8000,
            "cache_write_tokens": 3000,
            "cache_write_tokens_details": {
                "ttl_30m_tokens": 3000,
                "ttl_12h_tokens": 0,
            },
        },
    }
    p = PRICES["openai_gpt61_sol"]
    d = usage["input_tokens_details"]
    ordinary = usage["input_tokens"] - d["cached_tokens"] - d["cache_write_tokens"]
    cost = (
        usd(ordinary, p["input"])
        + usd(d["cached_tokens"], p["cache_read"])
        + usd(d["cache_write_tokens_details"]["ttl_30m_tokens"], p["cache_write"])
        + usd(d["cache_write_tokens_details"]["ttl_12h_tokens"], 5.00)
        + usd(usage["output_tokens"], p["output"])
    )
    return {"usage": usage, "ordinary_tokens": ordinary, "estimated_cost_usd": round(cost, 6)}


def main():
    results = {k: simulate(k) for k in PRICES}
    usage_demo = parse_usage_example()
    payload = {"results": results, "usage_field_demo": usage_demo, "fetched": "2026-10-30"}
    path = os.path.join(OUT, "ep23_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== 提示缓存成本模拟（12 轮 Agent）===")
    print(f"稳定前缀 {STABLE_PREFIX} + 每轮新增 {NEW_PER_TURN}，输出 {OUTPUT_PER_TURN}/轮\n")
    for k, r in results.items():
        s = r["summary"]
        print(f"{r['label']}")
        print(f"  无缓存: ${s['cost_without_cache_usd']:.4f}")
        print(f"  有缓存: ${s['cost_with_cache_usd']:.4f}")
        print(f"  节省:   ${s['savings_usd']:.4f} ({s['savings_pct']:.1f}%)")
        print(f"  缓存读 token 合计: {s['total_cached_tokens']:,}")
        print()
    print("--- usage 字段计费示例 (gpt-6.1-sol) ---")
    print(json.dumps(usage_demo, ensure_ascii=False, indent=2))
    print(f"\n结果已写入 {path}")


if __name__ == "__main__":
    main()
