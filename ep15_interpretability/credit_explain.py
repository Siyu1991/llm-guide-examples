"""可解释性入门实验（纯 numpy，虚构的信贷数据）：
1) 逻辑回归：系数本身就是解释（"白盒"）；
2) 小神经网络：用 梯度×输入、积分梯度（Integrated Gradients）、置换重要性 解释单个决策和整体；
3) 线性探针：从隐藏层里能不能"读出"某个没给模型的概念（这里是地区，即潜在的歧视代理变量）。"""
import json, os
import numpy as np
rng = np.random.default_rng(7)
FEATS = ["月收入(千元)", "负债率", "逾期次数", "工作年限", "近3月查询次数", "邮编区段"]

def make(n):
    region = rng.integers(0, 2, n)                     # 0/1 两个地区，模型输入里没有这一列
    income = rng.normal(12, 4, n) - 3 * region          # 地区与收入相关
    debt = np.clip(rng.normal(0.4, 0.15, n), 0, 1); overdue = rng.poisson(0.6, n); years = rng.integers(0, 20, n)
    inq = rng.poisson(2, n); zipc = region * 5 + rng.normal(0, 1, n)  # 邮编区段：几乎就是地区的代理
    logit = 0.35 * income - 6 * debt - 1.1 * overdue + 0.08 * years - 0.3 * inq - 0.25 * zipc - 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)
    X = np.c_[income, debt, overdue, years, inq, zipc]; return X, y, region

X, y, region = make(20000); Xt, yt, rt = make(5000)
mu, sd = X.mean(0), X.std(0); Z, Zt = (X - mu) / sd, (Xt - mu) / sd
sig = lambda z: 1 / (1 + np.exp(-z))

def train_lr(Z, y, epochs=300, lr=0.5):
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(epochs):
        p = sig(Z @ w + b); w -= lr * Z.T @ (p - y) / len(y); b -= lr * (p - y).mean()
    return w, b

def train_mlp(Z, y, h=32, epochs=200, lr=0.3):
    W1 = rng.normal(0, .5, (Z.shape[1], h)); b1 = np.zeros(h); w2 = rng.normal(0, .3, h); b2 = 0.0
    for _ in range(epochs):
        H = np.tanh(Z @ W1 + b1); p = sig(H @ w2 + b2); g = (p - y) / len(y)
        dW1 = Z.T @ (np.outer(g, w2) * (1 - H ** 2)); db1 = (np.outer(g, w2) * (1 - H ** 2)).sum(0)
        w2 -= lr * H.T @ g; b2 -= lr * g.sum(); W1 -= lr * dW1; b1 -= lr * db1
    return W1, b1, w2, b2

def mlp_f(P, Z): W1, b1, w2, b2 = P; H = np.tanh(Z @ W1 + b1); return sig(H @ w2 + b2), H
def mlp_grad(P, z):  # d p / d z，单个样本
    W1, b1, w2, b2 = P; h = np.tanh(z @ W1 + b1); p = sig(h @ w2 + b2); return p * (1 - p) * (W1 @ (w2 * (1 - h ** 2)))
def integrated_gradients(P, z, base, steps=64):
    alphas = (np.arange(steps) + 0.5) / steps
    return (z - base) * np.mean([mlp_grad(P, base + a * (z - base)) for a in alphas], 0)
def permutation_importance(f, Z, y, reps=5):
    base = ((f(Z) > .5) == y).mean(); out = []
    for j in range(Z.shape[1]):
        d = []
        for _ in range(reps):
            Zp = Z.copy(); Zp[:, j] = rng.permutation(Zp[:, j]); d.append(base - ((f(Zp) > .5) == y).mean())
        out.append(float(np.mean(d)))
    return out

def probe(H, r, Ht, rt):
    w, b = train_lr((H - H.mean(0)) / (H.std(0) + 1e-9), r)
    return float(((sig(((Ht - H.mean(0)) / (H.std(0) + 1e-9)) @ w + b) > .5) == rt).mean())

if __name__ == "__main__":
    w, b = train_lr(Z, y); acc_lr = float(((sig(Zt @ w + b) > .5) == yt).mean())
    P = train_mlp(Z, y); acc_mlp = float(((mlp_f(P, Zt)[0] > .5) == yt).mean())
    print(f"测试集准确率：逻辑回归 {acc_lr:.1%}，小神经网络 {acc_mlp:.1%}")
    print("1) 逻辑回归系数（标准化后，正=更容易通过）：", {f: round(float(c), 2) for f, c in zip(FEATS, w)})
    i = int(np.argmin(mlp_f(P, Zt)[0])); z = Zt[i]; base = np.zeros_like(z)
    ig = integrated_gradients(P, z, base); gx = mlp_grad(P, z) * (z - base)
    p_i = float(mlp_f(P, z[None])[0][0]); p_b = float(mlp_f(P, base[None])[0][0])
    print(f"2) 被拒申请人 #{i}：通过概率 {p_i:.3f}（基准「平均申请人」 {p_b:.3f}）")
    print("   原始特征：", {f: round(float(v), 2) for f, v in zip(FEATS, Xt[i])})
    print("   积分梯度：", {f: round(float(v), 3) for f, v in zip(FEATS, ig)}, f"合计 {ig.sum():.3f}，应≈{p_i - p_b:.3f}")
    print("   梯度×输入：", {f: round(float(v), 3) for f, v in zip(FEATS, gx)})
    pi = permutation_importance(lambda Z_: mlp_f(P, Z_)[0], Zt, yt)
    print("   置换重要性（打乱该列后准确率下降）：", {f: f"{v:.1%}" for f, v in zip(FEATS, pi)})
    _, H = mlp_f(P, Z); _, Ht = mlp_f(P, Zt)
    pr = probe(H, region.astype(float), Ht, rt.astype(float))
    P2 = train_mlp(Z[:, :5], y); _, H2 = mlp_f(P2, Z[:, :5]); _, H2t = mlp_f(P2, Zt[:, :5])
    pr2 = probe(H2, region.astype(float), H2t, rt.astype(float))
    acc2 = float(((mlp_f(P2, Zt[:, :5])[0] > .5) == yt).mean())
    print(f"3) 线性探针从隐藏层读出'地区'的准确率：含邮编 {pr:.1%}；去掉邮编后 {pr2:.1%}（去掉邮编后模型准确率 {acc2:.1%}）")
    pass_r = [float((mlp_f(P, Zt)[0][rt == g] > .5).mean()) for g in [0, 1]]
    pass_r2 = [float((mlp_f(P2, Zt[:, :5])[0][rt == g] > .5).mean()) for g in [0, 1]]
    print(f"   两地区通过率：含邮编 {pass_r[0]:.1%} vs {pass_r[1]:.1%}；去掉邮编 {pass_r2[0]:.1%} vs {pass_r2[1]:.1%}")
    os.makedirs("outputs", exist_ok=True)
    json.dump(dict(acc_lr=acc_lr, acc_mlp=acc_mlp, lr_coef=dict(zip(FEATS, map(float, w))), ig=dict(zip(FEATS, map(float, ig))),
                   gx=dict(zip(FEATS, map(float, gx))), perm=dict(zip(FEATS, pi)), probe=pr, probe_nozip=pr2, acc_nozip=acc2,
                   pass_rate=pass_r, pass_rate_nozip=pass_r2, x=dict(zip(FEATS, map(float, Xt[i]))), p=p_i, p_base=p_b),
              open("outputs/results.json", "w"), ensure_ascii=False, indent=1)
