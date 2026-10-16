"""延伸实验：同一位被拒申请人，换方法、换基准、换步数，解释会怎么变。
1) 精确 Shapley（6 个特征共 64 种组合，缺席特征用基准值代替）与积分梯度对比；
2) 三种基准：平均申请人 / 刚好通过的申请人 / 通过者平均；
3) 积分梯度步数与完备性误差；
4) 反事实：只改一个特征，要改到多少才能通过。纯 numpy，数据为程序生成。"""
import itertools, json, math, os
import numpy as np
import credit_explain as ce

def shapley(f, z, base):
    d = len(z); phi = np.zeros(d)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        for r in range(d):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(d - r - 1) / math.factorial(d)
                x = base.copy(); x[list(S)] = z[list(S)]; v0 = f(x)
                x[j] = z[j]; phi[j] += w * (f(x) - v0)
    return phi

def main():
    P = ce.train_mlp(ce.Z, ce.y)
    f = lambda x: float(ce.mlp_f(P, x[None])[0][0])
    pt = ce.mlp_f(P, ce.Zt)[0]
    i = int(np.argmin(pt)); z = ce.Zt[i]
    near = ce.Zt[(pt > 0.5) & (pt < 0.55)].mean(0)       # 刚好通过的申请人（平均）
    appr = ce.Zt[pt > 0.5].mean(0)                       # 通过者平均
    bases = {"平均申请人": np.zeros_like(z), "刚好通过的申请人": near, "通过者平均": appr}
    out = {"p": f(z), "bases": {}}
    for name, b in bases.items():
        ig = ce.integrated_gradients(P, z, b, steps=256); sh = shapley(f, z, b)
        out["bases"][name] = dict(p_base=f(b), ig=dict(zip(ce.FEATS, map(float, ig))), shap=dict(zip(ce.FEATS, map(float, sh))),
                                  ig_sum=float(ig.sum()), shap_sum=float(sh.sum()))
        print(f"基准={name} p_base={f(b):.3f}  目标差={f(z)-f(b):+.3f}")
        for k, n in enumerate(ce.FEATS): print(f"   {n:10s} IG {ig[k]:+.3f}  Shapley {sh[k]:+.3f}")
    out["steps"] = {}
    for s in [2, 4, 8, 16, 64, 256]:
        e = abs(ce.integrated_gradients(P, z, bases["平均申请人"], steps=s).sum() - (f(z) - f(bases["平均申请人"])))
        out["steps"][s] = float(e); print(f"IG 步数 {s:3d}: 完备性误差 {e:.4f}")
    # 反事实：单独改一个特征（原始单位），找最小改动使通过概率 >= 0.5
    x0 = ce.Xt[i].copy(); out["cf"] = {}
    for j, grid in [(0, np.arange(x0[0], 60, 0.1)), (1, np.arange(x0[1], -0.001, -0.01)), (2, np.arange(x0[2], -1, -1))]:
        hit = None
        for v in grid:
            x = x0.copy(); x[j] = v
            if f((x - ce.mu) / ce.sd) >= 0.5: hit = float(v); break
        out["cf"][ce.FEATS[j]] = hit; print(f"反事实：只改 {ce.FEATS[j]}，从 {x0[j]:.2f} 到 {hit}")
    os.makedirs("outputs", exist_ok=True)
    json.dump(out, open("outputs/shapley_vs_ig.json", "w"), ensure_ascii=False, indent=1)
    return out

if __name__ == "__main__":
    main()
