"""延伸模拟：哪一个旋钮最值钱？在 10 步任务上分别扫描
1) 每步成功率 p（换更强的模型 / 改工具设计）；2) 校验发现率 p_detect（校验写得好不好）；3) 步数 n（把固定流程改成代码）。
统一用"校验 + 最多重试 2 次"的策略，并把静默出错按每单 LOSS_SILENT 美元计入总成本。所有参数均为假设值。"""
import json, os, random
import simulate as S
LOSS_SILENT = 20.0

def run(n, p, p_detect, trials=20000, seed=0, retry=2, step_cost=S.COST_STEP):
    rnd = random.Random(seed); succ = silent = hand = 0; cost = 0.0
    old = S.COST_STEP; S.COST_STEP = step_cost
    try:
        for _ in range(trials):
            ok, c, h = S.episode(rnd, n, p, retry, True, p_detect); cost += c; succ += ok; hand += h; silent += (not ok and not h)
    finally:
        S.COST_STEP = old
    total = cost + hand * S.COST_HUMAN + silent * LOSS_SILENT
    return dict(success=succ / trials, silent=silent / trials, handoff=hand / trials, model_cost=cost / trials, total_per_task=total / trials)

if __name__ == "__main__":
    out = {"p": {}, "detect": {}, "n": {}, "cheap_vs_strong": {}}
    for p in [0.90, 0.95, 0.98, 0.99]:
        r = run(10, p, 0.9); out["p"][p] = r; print(f"p={p:.2f}: 成功 {r['success']:.1%} 静默 {r['silent']:.1%} 每单总成本 ${r['total_per_task']:.3f}")
    for d in [0.5, 0.7, 0.9, 0.99]:
        r = run(10, 0.95, d); out["detect"][d] = r; print(f"发现率={d:.2f}: 成功 {r['success']:.1%} 静默 {r['silent']:.1%} 每单总成本 ${r['total_per_task']:.3f}")
    for n in [4, 6, 10, 20]:
        r = run(n, 0.95, 0.9); out["n"][n] = r; print(f"n={n}: 成功 {r['success']:.1%} 静默 {r['silent']:.1%} 每单总成本 ${r['total_per_task']:.3f}")
    # 便宜模型（每步 $0.0005，p=0.93） vs 强模型（每步 $0.004，p=0.98）
    for name, c, p in [("便宜模型", 0.0005, 0.93), ("强模型", 0.004, 0.98)]:
        r = run(10, p, 0.9, step_cost=c); out["cheap_vs_strong"][name] = r
        print(f"{name}: 模型费 ${r['model_cost']:.4f} 成功 {r['success']:.1%} 静默 {r['silent']:.1%} 每单总成本 ${r['total_per_task']:.3f}")
    os.makedirs("outputs", exist_ok=True); json.dump(out, open("outputs/sweep.json", "w"), ensure_ascii=False, indent=1)
