"""延伸实验：GPTQ 的核心思想（误差补偿）用 numpy 写一遍，和"就近取整"(RTN)比较。
一个 512 输入 × 256 输出的线性层，校准数据的输入通道之间有相关性，且有少数离群通道。
评价指标看的是"层输出"的误差，而不是权重本身的误差，因为模型真正关心的是输出。"""
import json, os
import numpy as np

def scales(W, bits):  # 每个输出行一个缩放系数（对称）
    qmax = 2 ** (bits - 1) - 1
    return np.abs(W).max(1, keepdims=True) / qmax, qmax

def rtn(W, bits):
    s, qmax = scales(W, bits); return np.clip(np.round(W / s), -qmax - 1, qmax) * s

def gptq(W, X, bits, damp=0.01):
    W = W.copy(); s, qmax = scales(W, bits); d = W.shape[1]
    H = X.T @ X; H += damp * np.mean(np.diag(H)) * np.eye(d)
    U = np.linalg.cholesky(np.linalg.inv(H)).T          # 上三角，Hinv = U^T U
    Q = np.zeros_like(W)
    for i in range(d):
        q = np.clip(np.round(W[:, i:i+1] / s), -qmax - 1, qmax) * s
        Q[:, i:i+1] = q
        err = (W[:, i:i+1] - q) / U[i, i]
        W[:, i+1:] -= err @ U[i:i+1, i+1:]               # 把本列误差摊到还没量化的列上
    return Q

def out_err(W, Wq, X): return float(np.linalg.norm(X @ (W - Wq).T) / np.linalg.norm(X @ W.T))

def main(seed=0):
    rng = np.random.default_rng(seed); din, dout = 512, 256
    A = rng.normal(0, 1, (din, din)) / np.sqrt(din) + np.eye(din)   # 让输入通道相关
    def data(n):
        X = rng.normal(0, 1, (n, din)) @ A; X[:, :8] *= 20; return X  # 8 个离群通道
    Xc, Xt = data(1024), data(2048)                                   # 校准集 / 测试集
    W = rng.normal(0, 0.05, (dout, din))
    res = {}
    for b in [8, 4, 3, 2]:
        r = out_err(W, rtn(W, b), Xt); g = out_err(W, gptq(W, Xc, b), Xt)
        res[b] = dict(rtn=r, gptq=g); print(f"{b} bit  RTN 输出误差 {r:.2%}   GPTQ {g:.2%}   降低 {1 - g / r:.0%}")
    os.makedirs("outputs", exist_ok=True); json.dump(res, open("outputs/gptq_mini.json", "w"), indent=1)
    return res

if __name__ == "__main__":
    main()
