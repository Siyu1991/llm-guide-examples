#!/usr/bin/env python3
"""KV cache 显存估算 + API vs 自建保本点（离线）。

GPU 租价取自 Runpod 公开页（2026-10-30 查阅）：
  A100 PCIe Community Cloud $1.19/hr
  H100 PCIe Community Cloud $1.19→实际 $1.99/hr
  RTX 4090 Community Cloud $0.34/hr
API 对照：OpenAI gpt-6.1-sol 输入 $2 / 输出 $10 per MTok（同日官方价目）。
"""
from __future__ import annotations
import json, os, math
from dataclasses import asdict, dataclass

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

# 字节：每层每 token 的 K+V；fp16=2 bytes
def kv_bytes_per_token(n_layers: int, n_kv_heads: int, head_dim: int, bytes_per: int = 2) -> int:
    # K and V each: n_kv_heads * head_dim * bytes
    return n_layers * 2 * n_kv_heads * head_dim * bytes_per


@dataclass
class ModelCfg:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int


MODELS = [
    ModelCfg("llama-like-7b", 32, 8, 128),   # GQA 示意
    ModelCfg("llama-like-70b", 80, 8, 128),
]


def estimate_kv_gib(cfg: ModelCfg, batch: int, seq: int) -> float:
    bpt = kv_bytes_per_token(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
    return batch * seq * bpt / (1024 ** 3)


# Runpod Community Cloud（2026-10-30）
GPU_HOURLY = {
    "A100_PCIe": 1.19,
    "H100_PCIe": 1.99,
    "RTX_4090": 0.34,
}

API = {"input_per_mtok": 2.0, "output_per_mtok": 10.0}


def breakeven(gpu: str, tokens_out_per_hour: float, in_out_ratio: float = 2.0) -> dict:
    """给定自建每小时可产出的输出 token，算等价 API 费用与保本利用率。"""
    hourly = GPU_HOURLY[gpu]
    out_mtok = tokens_out_per_hour / 1e6
    in_mtok = out_mtok * in_out_ratio
    api_cost = in_mtok * API["input_per_mtok"] + out_mtok * API["output_per_mtok"]
    # 保本：api_cost >= hourly → 至少需要的输出 token/小时
    # hourly = out_m*(ratio*in_price + out_price) → out_m = hourly / (ratio*in + out)
    denom = in_out_ratio * API["input_per_mtok"] + API["output_per_mtok"]
    need_out_mtok = hourly / denom
    return {
        "gpu": gpu,
        "gpu_hourly_usd": hourly,
        "assumed_out_tokens_per_hour": tokens_out_per_hour,
        "api_equivalent_usd": round(api_cost, 4),
        "self_host_cheaper": api_cost > hourly,
        "breakeven_out_tokens_per_hour": int(need_out_mtok * 1e6),
        "notes": "忽略电费外的人力/闲置；tokens_out_per_hour 为假设吞吐",
    }


def main():
    kv_rows = []
    for cfg in MODELS:
        for batch, seq in [(1, 4096), (8, 4096), (32, 2048)]:
            gib = estimate_kv_gib(cfg, batch, seq)
            kv_rows.append({
                "model": cfg.name, "batch": batch, "seq": seq,
                "kv_gib": round(gib, 3),
                "bytes_per_token": kv_bytes_per_token(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim),
            })
    # 假设：单卡持续解码 80k output tok/hour（示意，非实测）
    breaks = [breakeven(g, 80_000) for g in GPU_HOURLY]
    breaks_hi = [breakeven(g, 400_000) for g in GPU_HOURLY]
    out = {
        "fetched": "2026-10-30",
        "gpu_hourly_runpod_community": GPU_HOURLY,
        "api_gpt61_sol": API,
        "kv_estimates": kv_rows,
        "breakeven_at_80k_out_tok_per_h": breaks,
        "breakeven_at_400k_out_tok_per_h": breaks_hi,
    }
    path = os.path.join(OUT, "ep28_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=== KV 显存估算（GiB）===")
    for r in kv_rows:
        print(f"{r['model']} b={r['batch']} s={r['seq']}: {r['kv_gib']} GiB")
    print("=== 保本（假设 80k / 400k out tok/h）===")
    for b in breaks:
        print(f"80k {b['gpu']}: GPU ${b['gpu_hourly_usd']}/h API≈${b['api_equivalent_usd']} cheaper={b['self_host_cheaper']} BE_out={b['breakeven_out_tokens_per_hour']}")
    for b in breaks_hi:
        print(f"400k {b['gpu']}: API≈${b['api_equivalent_usd']} cheaper={b['self_host_cheaper']}")
    print("写入", path)


if __name__ == "__main__":
    main()
