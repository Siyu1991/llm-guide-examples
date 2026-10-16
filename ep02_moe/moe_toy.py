"""第 2 期配套：用 numpy 写一个最小 MoE 层，实测三件事。
1) 同样"总参数"的稠密 FFN 与 MoE（top-k）在 CPU 上的耗时差；
2) 路由不均衡（路由坍塌）现象，以及 DeepSeek-V3 式"只调偏置"的无辅助损失均衡；
3) 按总参数 / 激活参数估算显存与单流解码速度上限。
只依赖 numpy，CPU 运行，内存占用 < 500MB。
"""
import json, time, argparse
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

class DenseFFN:
    def __init__(self, d, h, rng):
        self.w1 = (rng.standard_normal((d, h)) / np.sqrt(d)).astype(np.float32)
        self.w2 = (rng.standard_normal((h, d)) / np.sqrt(h)).astype(np.float32)
    def __call__(self, x):
        return gelu(x @ self.w1) @ self.w2
    @property
    def params(self):
        return self.w1.size + self.w2.size

class MoE:
    """n_experts 个专家 FFN + 线性路由器，每个 token 选 top_k 个专家。"""
    def __init__(self, d, h, n_experts, top_k, rng):
        self.experts = [DenseFFN(d, h, rng) for _ in range(n_experts)]
        self.router = (rng.standard_normal((d, n_experts)) / np.sqrt(d)).astype(np.float32)
        self.bias = np.zeros(n_experts, dtype=np.float32)  # 只用于选专家，不参与加权
        self.k = top_k
    def route(self, x):
        logits = x @ self.router
        scores = 1 / (1 + np.exp(-logits))           # DeepSeek-V3 用 sigmoid 亲和度
        sel = np.argsort(-(scores + self.bias), axis=1)[:, : self.k]
        g = np.take_along_axis(scores, sel, axis=1)
        g = g / g.sum(axis=1, keepdims=True)          # 门控权重归一化
        return sel, g
    def __call__(self, x):
        sel, g = self.route(x)
        out = np.zeros_like(x)
        for e, ex in enumerate(self.experts):         # 按专家分组计算：只算分到它的 token
            rows, slot = np.nonzero(sel == e)
            if rows.size:
                out[rows] += g[rows, slot, None] * ex(x[rows])
        return out, sel
    @property
    def params_total(self):
        return sum(e.params for e in self.experts) + self.router.size
    @property
    def params_active(self):
        return self.k * self.experts[0].params + self.router.size

def bench(fn, x, reps=5):
    fn(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(x); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def clustered_tokens(rng, n, d, n_clusters=6, skew=True):
    """模拟真实文本：token 隐藏向量聚成几簇，且簇的频率不均（像中文语料里常用字多）。"""
    centers = rng.standard_normal((n_clusters, d)).astype(np.float32) * 2.0
    p = np.array([0.4, 0.25, 0.15, 0.1, 0.06, 0.04]) if skew else np.ones(n_clusters) / n_clusters
    idx = rng.choice(n_clusters, size=n, p=p)
    return (centers[idx] + rng.standard_normal((n, d)).astype(np.float32)).astype(np.float32)

def load_stats(sel, n_experts):
    cnt = np.bincount(sel.ravel(), minlength=n_experts)
    return cnt, float(cnt.max() / cnt.mean()), int((cnt == 0).sum())

def balance_by_bias(moe, x, steps=60, gamma=0.02):
    """DeepSeek-V3 报告 2.1.2 节思路：每步后过载专家偏置 -gamma，空闲专家 +gamma。"""
    hist = []
    for _ in range(steps):
        sel, _ = moe.route(x)
        cnt, ratio, dead = load_stats(sel, len(moe.experts))
        hist.append(ratio)
        moe.bias -= gamma * np.sign(cnt - cnt.mean()).astype(np.float32)
    return hist

def capacity(total_b, active_b, bw_tbps):
    rows = []
    for name, bytes_per in [("BF16", 2), ("FP8/INT8", 1), ("4-bit", 0.5), ("2-bit", 0.25)]:
        mem = total_b * bytes_per
        rows.append({"精度": name, "权重显存GB": round(mem, 1),
                     "MoE单流上限tok/s": round(bw_tbps * 1000 / (active_b * bytes_per), 1),
                     "若当稠密读全部tok/s": round(bw_tbps * 1000 / mem, 2)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="小尺寸快速自检")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    d, h, E, K, T = (128, 256, 8, 2, 256) if a.quick else (512, 1024, 16, 2, 2048)
    x = rng.standard_normal((T, d)).astype(np.float32)
    moe = MoE(d, h, E, K, rng)
    dense_same_total = DenseFFN(d, h * E, rng)      # 总参数相同
    dense_same_active = DenseFFN(d, h * K, rng)     # 激活参数相同
    t_moe = bench(lambda z: moe(z), x)
    t_big = bench(dense_same_total, x)
    t_small = bench(dense_same_active, x)
    res = {"config": dict(d_model=d, expert_hidden=h, n_experts=E, top_k=K, tokens=T),
           "params": dict(moe_total=moe.params_total, moe_active=moe.params_active,
                          dense_same_total=dense_same_total.params, dense_same_active=dense_same_active.params),
           "time_ms": dict(moe=round(t_moe * 1e3, 2), dense_same_total=round(t_big * 1e3, 2),
                           dense_same_active=round(t_small * 1e3, 2))}
    # 正确性：MoE 输出与"逐 token 手算 top-k 加权"一致
    out, sel = moe(x[:4])
    _, g = moe.route(x[:4])
    ref = np.stack([sum(g[i, j] * moe.experts[sel[i, j]](x[i:i + 1])[0] for j in range(K)) for i in range(4)])
    assert np.allclose(out, ref, atol=1e-4), "MoE 输出与手算不一致"
    # 负载均衡
    xs = clustered_tokens(rng, T, d)
    moe2 = MoE(d, h, E, K, np.random.default_rng(1))
    cnt0, r0, dead0 = load_stats(moe2.route(xs)[0], E)
    hist = balance_by_bias(moe2, xs, steps=30 if a.quick else 80)
    cnt1, r1, dead1 = load_stats(moe2.route(xs)[0], E)
    res["balance"] = dict(before_counts=cnt0.tolist(), before_max_over_mean=round(r0, 2), before_idle=dead0,
                          after_counts=cnt1.tolist(), after_max_over_mean=round(r1, 2), after_idle=dead1,
                          history=[round(v, 3) for v in hist])
    assert r1 < r0, "偏置均衡应降低最大负载比"
    # 部署估算：Muse Spark Open 816B/42B，Mac Studio M5 Ultra 带宽 1.2TB/s（Meta 博客公开数字）
    res["capacity_muse_spark_open"] = capacity(816, 42, 1.2)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open("result.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nMoE 总参数是激活参数的 {moe.params_total / moe.params_active:.1f} 倍；"
          f"耗时：MoE {t_moe*1e3:.1f}ms，同总参数稠密 {t_big*1e3:.1f}ms，同激活参数稠密 {t_small*1e3:.1f}ms")
    import resource
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
