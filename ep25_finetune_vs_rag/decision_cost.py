#!/usr/bin/env python3
"""微调 / RAG / 纯提示词：离线成本与决策计算器。

价格（2026-10-30 查阅官方文档，美元）：
  OpenAI 微调 gpt-4.1（现有客户可训练至 2027-01-06（OpenAI deprecations））：
    训练 25.00 / 百万训练 token
    推理输入 3.00 / 百万，输出 12.00 / 百万
    https://developers.openai.com/api/docs/pricing
  OpenAI gpt-6.1-sol Standard（对比基线，短上下文）：
    输入 2.00，输出 10.00 / 百万
  嵌入 text-embedding-3-large：0.13 / 百万 token
  Anthropic Claude Sonnet 5.5：输入 2.00，输出 10.00
  假设自建向量库月费（托管）：30 USD（可改）

本脚本不调用 API。决策规则是启发式打分，不是论文结论复刻。
论文引用见 README：Ovadia et al. 2023；Balaguer et al. 2024；Hu et al. 2021 LoRA。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

PRICES = {
    "ft_train_per_mtok": 25.0,       # gpt-4.1 fine-tuning training
    "ft_input_per_mtok": 3.0,
    "ft_output_per_mtok": 12.0,
    "base_input_per_mtok": 2.0,      # gpt-6.1-sol
    "base_output_per_mtok": 10.0,
    "embed_per_mtok": 0.13,          # text-embedding-3-large
    "vector_db_monthly_usd": 30.0,
}


@dataclass
class Scenario:
    name: str
    # 知识是否频繁变更
    knowledge_churn: str  # low / medium / high
    # 是否需要固定风格/格式
    style_lock: bool
    # 每月问答次数
    monthly_queries: int
    # 每次问答：提示词 token（含检索拼接）、输出 token
    prompt_tokens: int
    output_tokens: int
    # RAG：语料总 token、每次检索拼入的 chunk token、每月重建嵌入比例
    corpus_tokens: int
    retrieved_tokens: int
    reembed_fraction_per_month: float
    # 微调：训练样本数、每样本 token、训练 epoch
    ft_examples: int
    ft_tokens_per_example: int
    ft_epochs: int
    # 纯提示：系统提示额外 token（把说明塞进 context）
    prompt_only_extra_tokens: int


def cost_prompt_only(s: Scenario, p: dict) -> dict:
    inp = (s.prompt_tokens + s.prompt_only_extra_tokens) * s.monthly_queries
    out = s.output_tokens * s.monthly_queries
    monthly = inp * p["base_input_per_mtok"] / 1e6 + out * p["base_output_per_mtok"] / 1e6
    return {
        "setup_usd": 0.0,
        "monthly_usd": round(monthly, 4),
        "year1_usd": round(monthly * 12, 4),
        "notes": "无训练费；上下文更长，命中率依赖提示质量",
    }


def cost_rag(s: Scenario, p: dict) -> dict:
    setup_embed = s.corpus_tokens * p["embed_per_mtok"] / 1e6
    monthly_reembed = s.corpus_tokens * s.reembed_fraction_per_month * p["embed_per_mtok"] / 1e6
    # 每次请求：系统+问题+检索 chunk
    inp = (s.prompt_tokens + s.retrieved_tokens) * s.monthly_queries
    out = s.output_tokens * s.monthly_queries
    llm = inp * p["base_input_per_mtok"] / 1e6 + out * p["base_output_per_mtok"] / 1e6
    monthly = llm + monthly_reembed + p["vector_db_monthly_usd"]
    return {
        "setup_usd": round(setup_embed, 4),
        "monthly_usd": round(monthly, 4),
        "year1_usd": round(setup_embed + monthly * 12, 4),
        "breakdown": {
            "setup_embed_usd": round(setup_embed, 4),
            "monthly_llm_usd": round(llm, 4),
            "monthly_reembed_usd": round(monthly_reembed, 4),
            "monthly_vectordb_usd": p["vector_db_monthly_usd"],
        },
        "notes": "知识更新靠重建索引；向量库月费按托管假设",
    }


def cost_finetune(s: Scenario, p: dict) -> dict:
    train_tokens = s.ft_examples * s.ft_tokens_per_example * s.ft_epochs
    setup = train_tokens * p["ft_train_per_mtok"] / 1e6
    # 微调后提示可更短（假定比 prompt_only 少 extra，且无 retrieved）
    inp = s.prompt_tokens * s.monthly_queries
    out = s.output_tokens * s.monthly_queries
    monthly = inp * p["ft_input_per_mtok"] / 1e6 + out * p["ft_output_per_mtok"] / 1e6
    return {
        "setup_usd": round(setup, 4),
        "monthly_usd": round(monthly, 4),
        "year1_usd": round(setup + monthly * 12, 4),
        "train_tokens": train_tokens,
        "notes": "训练按 gpt-4.1 微调价；推理输入价高于同代基线，需单独评估",
    }


def decide(s: Scenario) -> dict:
    """启发式：高 churn → RAG；强 style + 低 churn → FT；小流量试错 → prompt。"""
    scores = {"prompt": 0, "rag": 0, "finetune": 0}
    if s.monthly_queries < 2000:
        scores["prompt"] += 2
    if s.knowledge_churn == "high":
        scores["rag"] += 3
        scores["finetune"] -= 2
    elif s.knowledge_churn == "medium":
        scores["rag"] += 2
        scores["prompt"] += 1
    else:
        scores["finetune"] += 1
        scores["prompt"] += 1
    if s.style_lock:
        scores["finetune"] += 2
        scores["prompt"] -= 1
    if s.corpus_tokens > 2_000_000:
        scores["rag"] += 1
    if s.ft_examples < 200:
        scores["finetune"] -= 2
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return {"scores": scores, "recommend": ranked[0][0], "ranked": ranked}


def run_scenario(s: Scenario) -> dict:
    p = PRICES
    return {
        "scenario": asdict(s),
        "costs": {
            "prompt": cost_prompt_only(s, p),
            "rag": cost_rag(s, p),
            "finetune": cost_finetune(s, p),
        },
        "decision": decide(s),
        "prices_used": p,
    }


DEFAULTS = [
    Scenario(
        name="客服 FAQ（知识常变）",
        knowledge_churn="high",
        style_lock=False,
        monthly_queries=50_000,
        prompt_tokens=800,
        output_tokens=300,
        corpus_tokens=5_000_000,
        retrieved_tokens=1_500,
        reembed_fraction_per_month=0.2,
        ft_examples=800,
        ft_tokens_per_example=600,
        ft_epochs=3,
        prompt_only_extra_tokens=2_000,
    ),
    Scenario(
        name="内部公文风格改写（知识稳）",
        knowledge_churn="low",
        style_lock=True,
        monthly_queries=8_000,
        prompt_tokens=600,
        output_tokens=400,
        corpus_tokens=200_000,
        retrieved_tokens=800,
        reembed_fraction_per_month=0.05,
        ft_examples=1_200,
        ft_tokens_per_example=700,
        ft_epochs=3,
        prompt_only_extra_tokens=1_200,
    ),
    Scenario(
        name="小流量试点（先验证）",
        knowledge_churn="medium",
        style_lock=False,
        monthly_queries=500,
        prompt_tokens=1_000,
        output_tokens=350,
        corpus_tokens=500_000,
        retrieved_tokens=1_200,
        reembed_fraction_per_month=0.1,
        ft_examples=100,
        ft_tokens_per_example=500,
        ft_epochs=3,
        prompt_only_extra_tokens=1_500,
    ),
]


def main():
    results = [run_scenario(s) for s in DEFAULTS]
    path = os.path.join(OUT, "ep25_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"fetched": "2026-10-30", "results": results}, f, ensure_ascii=False, indent=2)

    print("=== 微调 / RAG / 提示词 成本对比 ===")
    print(f"价格快照: OpenAI gpt-4.1 FT 训练 ${PRICES['ft_train_per_mtok']}/MTok, "
          f"FT 推理 ${PRICES['ft_input_per_mtok']}/${PRICES['ft_output_per_mtok']}; "
          f"基线 ${PRICES['base_input_per_mtok']}/${PRICES['base_output_per_mtok']}; "
          f"embed ${PRICES['embed_per_mtok']}/MTok\n")
    for r in results:
        print(f"## {r['scenario']['name']}")
        print(f"  推荐: {r['decision']['recommend']}  分数={r['decision']['scores']}")
        for k, c in r["costs"].items():
            print(f"  {k:8s}  setup=${c['setup_usd']:<8} monthly=${c['monthly_usd']:<8} year1=${c['year1_usd']}")
        print()
    print(f"写入 {path}")


if __name__ == "__main__":
    main()
