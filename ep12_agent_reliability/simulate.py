"""为什么 demo 很美、上线很难：每步成功率 p、任务需要 n 步时，整体成功率约 p^n。
蒙特卡洛模拟 4 种策略，算"每个成功任务的成本"。参数为假设值，可改。"""
import json, os, random
COST_STEP = 0.004   # 每步模型调用成本（美元，假设：约 1500 输入 + 200 输出 token，按 $2/$10 计）
COST_HUMAN = 2.0    # 转人工处理一单的成本（美元，假设）

def episode(rnd, n, p, retry, verify, p_detect=0.9):
    cost = 0.0
    for _ in range(n):
        tries = 1 + retry
        for t in range(tries):
            cost += COST_STEP; ok = rnd.random() < p
            if verify: cost += COST_STEP * 0.3  # 校验步骤（规则或小模型），按三成成本计
            if ok: break
            if not verify or rnd.random() > p_detect: return False, cost, False  # 没校验/没发现：带错前进 = 失败且无人知道
        else:
            return False, cost, True  # 发现失败，转人工
    return True, cost, False

def run(n, p, retry, verify, trials=20000, seed=0):
    rnd = random.Random(seed); succ = silent = handoff = 0; cost = 0.0
    for _ in range(trials):
        ok, c, h = episode(rnd, n, p, retry, verify); cost += c
        succ += ok; handoff += h; silent += (not ok and not h)
    tot = cost + handoff * COST_HUMAN
    return dict(success=succ / trials, silent_fail=silent / trials, handoff=handoff / trials,
                cost_per_task=cost / trials, cost_per_success=tot / max(1, succ))

if __name__ == "__main__":
    out = {}
    print("每步成功率 p 与步数 n 的理论整体成功率 p^n：")
    for p in [0.99, 0.95, 0.9]: print(f"  p={p}: " + "  ".join(f"n={n}:{p**n:.0%}" for n in [1, 5, 10, 20]))
    print("\n策略对比（n=10 步，p=0.95）：")
    for name, r, v in [("裸跑", 0, False), ("只重试", 2, False), ("只校验", 0, True), ("校验+重试", 2, True)]:
        res = run(10, 0.95, r, v); out[name] = res
        print(f"  {name:6s} 成功 {res['success']:.1%}  静默出错 {res['silent_fail']:.1%}  转人工 {res['handoff']:.1%}  "
              f"每单模型成本 ${res['cost_per_task']:.4f}  每个成功任务总成本(含人工) ${res['cost_per_success']:.3f}")
    curve = {n: run(n, 0.95, 2, True, 5000)["success"] for n in [1, 2, 5, 10, 15, 20, 30]}
    raw = {n: 0.95 ** n for n in curve}
    out["curve"] = {"steps": list(curve), "guarded": list(curve.values()), "raw": list(raw.values())}
    os.makedirs("outputs", exist_ok=True); json.dump(out, open("outputs/results.json", "w"), ensure_ascii=False, indent=1)
