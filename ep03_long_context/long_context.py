"""第 3 期配套：长上下文的三笔账。
1) 实测：单层因果自注意力的 prefill 耗时随长度 n 的增长（numpy，CPU），看 n² 的形状；
2) 计算：不同模型结构的 KV 缓存大小（MHA/GQA/MLA）；
3) 计算：合同审阅场景下"整份重发 / 整份+缓存 / 阶梯价 / 检索片段"四种方案的成本。
只依赖 numpy。所有价格是 2026-10-16 官方页面数字，使用前请自行核对。
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # 单线程，减少多核调度带来的计时抖动
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, time, argparse, resource
import numpy as np

def causal_attention(q, k, v, block=512):
    """分块计算的因果注意力（思路同 FlashAttention：不一次性存 n×n 矩阵，但计算量仍是 n²）。"""
    n, d = q.shape
    out = np.empty_like(v)
    scale = 1 / np.sqrt(d)
    for s in range(0, n, block):
        e = min(n, s + block)
        att = (q[s:e] @ k[:e].T) * scale
        mask = np.arange(e)[None, :] > np.arange(s, e)[:, None]
        att[mask] = -np.inf
        att -= att.max(axis=1, keepdims=True)
        np.exp(att, out=att)
        att /= att.sum(axis=1, keepdims=True)
        out[s:e] = att @ v[:e]
    return out

def bench_prefill(lengths, d_head=128, heads=8, reps=3):
    rng = np.random.default_rng(0)
    rows = []
    for n in lengths:
        q, k, v = (rng.standard_normal((n, d_head)).astype(np.float32) for _ in range(3))
        causal_attention(q[:64], k[:64], v[:64])
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            for _ in range(heads):
                causal_attention(q, k, v)
            ts.append(time.perf_counter() - t)
        rows.append({"n": n, "ms": round(float(min(ts)) * 1e3, 1)})   # 取最小值：最接近无干扰时的耗时
    return rows

def kv_bytes_per_token(layers, kv_heads=None, head_dim=None, mla_dim=None, bytes_per=2):
    if mla_dim:                      # MLA：每层只存一个压缩向量（DeepSeek-V3：512 + 64 = 576 维）
        return layers * mla_dim * bytes_per
    return 2 * layers * kv_heads * head_dim * bytes_per

MODELS = {   # 公开结构参数：Llama 3 论文表 3；Llama 2 7B 为 MHA 32 头；DeepSeek-V3 报告 2.1.1
    "Llama 2 7B（MHA）": dict(layers=32, kv_heads=32, head_dim=128),
    "Llama 3 70B（GQA）": dict(layers=80, kv_heads=8, head_dim=128),
    "DeepSeek-V3（MLA）": dict(layers=61, mla_dim=576),
}

def contract_cost(tokens=100_000, asks=20, out=1000, rag_tokens=8000):
    M = 1e6
    a = asks * tokens * 2 / M + asks * out * 10 / M                       # Argon 首发价 $2/$10
    b = tokens * 2 / M + (asks - 1) * tokens * 0.10 / M + asks * out * 10 / M  # 缓存读 $0.10
    c_hi = asks * tokens * 0.50 / M + asks * out * 2.50 / M               # Haiku 5.5 >10 万档
    c_lo = asks * 95_000 * 0.10 / M + asks * out * 0.50 / M               # 清洗到 9.5 万 token
    c_lo_cache = 95_000 * 0.125 / M + (asks - 1) * 95_000 * 0.01 / M + asks * out * 0.50 / M
    d = asks * rag_tokens * 2 / M + asks * out * 10 / M
    return {"A 整份重发(Argon)": a, "B 整份+缓存(Argon)": b, "C1 Haiku5.5 超10万档": c_hi,
            "C2 Haiku5.5 清洗到9.5万": c_lo, "C3 Haiku5.5 清洗+缓存": c_lo_cache, "D 检索8K片段(Argon)": d}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    lengths = [256, 512, 1024] if a.quick else [512, 1024, 2048, 4096, 8192]
    pre = bench_prefill(lengths)
    base = pre[0]
    for r in pre:
        r["相对耗时"] = round(r["ms"] / base["ms"], 1); r["长度倍数"] = r["n"] // base["n"]
    kv = {m: round(kv_bytes_per_token(**p) / 1024, 1) for m, p in MODELS.items()}
    kv_128k = {m: round(kv_bytes_per_token(**p) * 128_000 / 2**30, 1) for m, p in MODELS.items()}
    # 自检：Llama 3 70B 每 token 327,680 字节
    assert kv_bytes_per_token(**MODELS["Llama 3 70B（GQA）"]) == 327_680
    cost = {k: round(v, 3) for k, v in contract_cost().items()}
    res = {"prefill_single_layer_8heads": pre, "kv_KB_per_token": kv, "kv_GiB_at_128k": kv_128k,
           "contract_cost_usd_per_doc": cost}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open("result.json", "w"), ensure_ascii=False, indent=1)
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
