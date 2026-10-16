"""量化原理实验（纯 numpy）：
1) 对一个带离群值的权重矩阵做 per-tensor / per-channel / 分组（group=64/128）量化到 8/4/3/2 bit，比较误差；
2) 训练一个小 MLP 分类器，把权重量化后看准确率变化；
3) 按参数量 × 位宽估算显存（只算权重，另加 KV 缓存与运行开销）。"""
import json, os, time
import numpy as np
rng = np.random.default_rng(0)

def quant(w, bits, axis=None, group=None):
    """对称线性量化：q = round(w / s)，s = max|w| / (2^(bits-1)-1)；返回反量化后的矩阵。"""
    qmax = 2 ** (bits - 1) - 1
    if group:
        shp = w.shape; g = w.reshape(-1, group)
        s = np.abs(g).max(1, keepdims=True) / qmax; s[s == 0] = 1
        return (np.clip(np.round(g / s), -qmax - 1, qmax) * s).reshape(shp)
    s = np.abs(w).max(axis=axis, keepdims=axis is not None) / qmax
    return np.clip(np.round(w / s), -qmax - 1, qmax) * s

def rel_err(w, wq): return float(np.linalg.norm(w - wq) / np.linalg.norm(w))

def exp_matrix():
    w = rng.normal(0, 0.02, (1024, 1024)).astype(np.float32)
    w[:, rng.choice(1024, 8, replace=False)] *= 30   # 少数"离群通道"，真实大模型里很常见（LLM.int8 论文）
    res = {}
    for bits in [8, 4, 3, 2]:
        res[bits] = {"per-tensor": rel_err(w, quant(w, bits)), "per-channel": rel_err(w, quant(w, bits, axis=0)),
                     "group128": rel_err(w, quant(w, bits, group=128)), "group64": rel_err(w, quant(w, bits, group=64))}
    return res

def make_data(n):
    x = rng.normal(size=(n, 20)); y = ((np.sin(x[:, 0] * 2) + x[:, 1] * x[:, 2] - x[:, 3] ** 2 + 0.5 * x[:, 4]) > 0).astype(int)
    return x, y

def train_mlp(x, y, h=256, epochs=60, lr=0.05):
    W1 = rng.normal(0, 0.3, (20, h)); b1 = np.zeros(h); W2 = rng.normal(0, 0.1, (h, 2)); b2 = np.zeros(2)
    for _ in range(epochs):
        for i in range(0, len(x), 128):
            xb, yb = x[i:i+128], y[i:i+128]; hdn = np.maximum(0, xb @ W1 + b1); z = hdn @ W2 + b2
            p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True); p[np.arange(len(yb)), yb] -= 1; p /= len(yb)
            dW2 = hdn.T @ p; dh = p @ W2.T * (hdn > 0); dW1 = xb.T @ dh
            W2 -= lr * dW2; b2 -= lr * p.sum(0); W1 -= lr * dW1; b1 -= lr * dh.sum(0)
    return W1, b1, W2, b2

def acc(params, x, y):
    W1, b1, W2, b2 = params; return float(((np.maximum(0, x @ W1 + b1) @ W2 + b2).argmax(1) == y).mean())

def exp_mlp():
    xtr, ytr = make_data(20000); xte, yte = make_data(5000); P = train_mlp(xtr, ytr)
    out = {"fp32": acc(P, xte, yte)}
    for bits in [8, 4, 3, 2]:
        for mode in ["per-tensor", "per-channel"]:
            ax = None if mode == "per-tensor" else 0
            Q = (quant(P[0], bits, ax), P[1], quant(P[2], bits, ax), P[3]); out[f"int{bits}-{mode}"] = acc(Q, xte, yte)
    return out

def memory_gb(params_b, bits, overhead=1.1):
    """只算权重的显存/内存（GB），overhead 粗略计入量化缩放系数和运行时开销（假设 10%）。"""
    return params_b * 1e9 * bits / 8 / 1e9 * overhead

if __name__ == "__main__":
    t = time.time(); m = exp_matrix(); print("1) 1024x1024 带离群通道矩阵，相对误差 ‖W-Ŵ‖/‖W‖：")
    for b, r in m.items(): print(f"  {b}bit  " + "  ".join(f"{k}={v:.3%}" for k, v in r.items()))
    a = exp_mlp(); print("2) 小 MLP（20→256→2）测试集准确率：", {k: f"{v:.1%}" for k, v in a.items()})
    print("3) 只算权重的内存估算（GB）：")
    mem = {}
    for name, pb in [("1.5B", 1.5), ("7B", 7), ("32B", 32), ("70B", 70)]:
        mem[name] = {f"{b}bit": round(memory_gb(pb, b), 1) for b in [16, 8, 4.5, 2.5]}
        print(f"  {name}: " + "  ".join(f"{k}={v}" for k, v in mem[name].items()))
    print(f"总耗时 {time.time()-t:.1f}s")
    os.makedirs("outputs", exist_ok=True); json.dump({"matrix": m, "mlp": a, "mem": mem}, open("outputs/results.json", "w"), indent=1)
