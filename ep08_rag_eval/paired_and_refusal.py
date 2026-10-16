"""延伸实验（纯标准库）：
1) 配对比较：两个配置跑同一批题，只看"一个对一个错"的题，用精确 McNemar 检验判断差异是不是真的；
2) 裁判校准：模拟一个会犯错、偏爱长回答的"大模型裁判"，和规则裁判比一致率与 Cohen's kappa（裁判误差参数为假设值）；
3) 无答案题：知识库里查不到的问题，加一道"检索分数太低就拒答"的闸，看拒答率和误拒率怎么取舍。"""
import json, math, os, random
from data import DOCS, QA
from rag_eval import BM25, grams, run

NOANS = ["你们支持花呗分期吗", "可以开发票抬头写个人吗", "店铺在哪个城市有仓库", "买的家电坏了能上门维修吗",
         "能用美元付款吗", "你们招聘客服吗"]  # 知识库里没有答案的问题（作者编写）

def mcnemar_exact(a, b):
    """a、b 为两配置每题是否答对的 0/1 列表。返回 (只有A对, 只有B对, 双侧 p 值)。"""
    n10 = sum(1 for x, y in zip(a, b) if x and not y); n01 = sum(1 for x, y in zip(a, b) if y and not x); n = n10 + n01
    if n == 0: return n10, n01, 1.0
    k = min(n10, n01); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n10, n01, min(1.0, 2 * p)

def kappa(x, y):
    n = len(x); po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n; pe = px * py + (1 - px) * (1 - py)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0

def noisy_judge(rows, flip=0.08, long_bias=0.25, seed=0):
    """模拟裁判：以 flip 概率判反；对长度超过 30 字的错误回答，再以 long_bias 概率误判为对。"""
    rnd = random.Random(seed); out = []
    for r in rows:
        v = int(r["correct"])
        if rnd.random() < flip: v = 1 - v
        if v == 0 and len(r["answer"]) > 30 and rnd.random() < long_bias: v = 1
        out.append(v)
    return out

def best_score(bm, q):
    s_max = 0.0
    for i in bm.ids:
        s = 0.0
        for g in set(grams(q)):
            f = bm.tf[i].get(g, 0)
            if f: s += bm.idf[g] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * bm.len[i] / bm.avg))
        s_max = max(s_max, s)
    return s_max

def main():
    out = {}
    _, A = run(1, "overlap"); _, B = run(3, "bm25")
    a = [int(r["correct"]) for r in A]; b = [int(r["correct"]) for r in B]
    n10, n01, p = mcnemar_exact(a, b); out["mcnemar"] = dict(only_overlap1=n10, only_bm25_3=n01, p=p)
    print(f"1) 字重合 k=1 vs BM25 k=3：只有前者对 {n10} 题，只有后者对 {n01} 题，McNemar 精确检验 p={p:.3f}")
    # 同样的差距放大到 400 题（每题复制 16 份只是为了演示样本量的作用）
    n10b, n01b, pb = mcnemar_exact(a * 16, b * 16); out["mcnemar_x16"] = dict(only_overlap1=n10b, only_bm25_3=n01b, p=pb)
    print(f"   同样比例放大到 {len(a) * 16} 题：{n10b} vs {n01b}，p={pb:.2g}")
    out["judge"] = {}
    for flip, lb in [(0.0, 0.0), (0.05, 0.0), (0.08, 0.25), (0.15, 0.4)]:
        rows = B * 20  # 480 条判卷记录
        jx = noisy_judge(rows, flip, lb, seed=1); gold = [int(r["correct"]) for r in rows]
        po, kp = kappa(gold, jx); acc_j = sum(jx) / len(jx)
        out["judge"][f"{flip}_{lb}"] = dict(agree=po, kappa=kp, judged_acc=acc_j, true_acc=sum(gold) / len(gold))
        print(f"2) 裁判判反率 {flip:.0%}、长回答偏好 {lb:.0%}：一致率 {po:.1%}，kappa {kp:.2f}，裁判给出的答对率 {acc_j:.1%}（真实 {sum(gold)/len(gold):.1%}）")
    bm = BM25(DOCS); ans_s = [best_score(bm, q) for q, _, _ in QA]; no_s = [best_score(bm, q) for q in NOANS]
    out["refusal"] = {}
    for th in [0, 4, 6, 8, 10]:
        refuse = sum(s < th for s in no_s) / len(no_s); wrong_refuse = sum(s < th for s in ans_s) / len(ans_s)
        out["refusal"][th] = dict(refuse_noans=refuse, refuse_answerable=wrong_refuse)
        print(f"3) 检索最高分 < {th:>2}：无答案题拒答 {refuse:.0%}，有答案题被误拒 {wrong_refuse:.0%}")
    out["score_range"] = dict(answerable=[min(ans_s), max(ans_s)], noans=[min(no_s), max(no_s)])
    print(f"   有答案题最高分区间 {min(ans_s):.1f}–{max(ans_s):.1f}，无答案题 {min(no_s):.1f}–{max(no_s):.1f}")
    os.makedirs("outputs", exist_ok=True); json.dump(out, open("outputs/paired_and_refusal.json", "w"), ensure_ascii=False, indent=1)
    return out

if __name__ == "__main__":
    main()
