"""单 Agent vs 多 Agent（规划-编码-审查）对比。任务：按需求写 3 个函数并通过单元测试。
离线 Mock 模式下，"编码者"按固定概率写出有 bug 的版本（模拟模型出错），"审查者"真的运行单元测试并把报错发回。
有 OPENAI_API_KEY 时编码者改用真实模型。统计：通过率、模型调用次数、上下文字符量（≈成本）。"""
import json, os, random, re, sys, urllib.request, textwrap, traceback

TASKS = {
 "slugify": ("把字符串转成小写、空格换成连字符、去掉首尾空格", "assert slugify('  Hello World ')=='hello-world'\nassert slugify('A B  C')=='a-b--c'"),
 "median": ("返回列表中位数，偶数个取中间两数平均", "assert median([3,1,2])==2\nassert median([4,1,3,2])==2.5"),
 "dedupe": ("去重并保持原顺序", "assert dedupe([3,1,3,2,1])==[3,1,2]\nassert dedupe([])==[]"),
}
GOOD = {"slugify": "def slugify(s):\n    return s.strip().lower().replace(' ', '-')",
        "median": "def median(xs):\n    s = sorted(xs); n = len(s)\n    return s[n//2] if n % 2 else (s[n//2-1] + s[n//2]) / 2",
        "dedupe": "def dedupe(xs):\n    seen = set(); out = []\n    for x in xs:\n        if x not in seen: seen.add(x); out.append(x)\n    return out"}
BAD = {"slugify": "def slugify(s):\n    return s.lower().replace(' ', '-')",
       "median": "def median(xs):\n    s = sorted(xs)\n    return s[len(s)//2]",
       "dedupe": "def dedupe(xs):\n    return list(set(xs))"}

class Meter:
    def __init__(self): self.calls = 0; self.chars = 0
    def add(self, prompt, reply): self.calls += 1; self.chars += len(prompt) + len(reply)

def coder_mock(name, spec, feedback, rnd, p_bug):
    # 没收到报错时按 p_bug 概率出错；收到报错后修好的概率 0.8（Mock 假设）
    if feedback: return GOOD[name] if rnd.random() < 0.8 else BAD[name]
    return BAD[name] if rnd.random() < p_bug else GOOD[name]

def coder_llm(name, spec, feedback, rnd, p_bug):
    prompt = f"用 Python 写函数 {name}：{spec}。只输出代码。" + (f"\n上次代码测试失败：{feedback}\n请修复。" if feedback else "")
    body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    req = urllib.request.Request(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions", data=json.dumps(body).encode(),
          headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
    m = re.search(r"```(?:python)?\n(.*?)```", txt, re.S); return m.group(1) if m else txt

def run_tests(code, tests):
    ns = {}
    try: exec(code, ns); exec(tests, ns); return None
    except Exception as e: return f"{type(e).__name__}: {traceback.format_exc().splitlines()[-2].strip()}"

def single_agent(rnd, p_bug, coder, m):
    # 一个 Agent 一次写完 3 个函数，不运行测试，自己说"完成了"
    ok = 0; prompt = "需求：" + json.dumps({k: v[0] for k, v in TASKS.items()}, ensure_ascii=False)
    for name, (spec, tests) in TASKS.items():
        code = coder(name, spec, None, rnd, p_bug); m.add(prompt, code); prompt += code
        ok += run_tests(code, tests) is None   # 只在最后由我们验收
    return ok

def single_with_tests(rnd, p_bug, coder, m, max_rounds=3):
    # 还是一个 Agent，但每写完一个函数就由程序自动跑测试，把报错原样喂回（不额外调用模型）
    ok = 0
    for name, (spec, tests) in TASKS.items():
        feedback = None
        for _ in range(max_rounds):
            code = coder(name, spec, feedback, rnd, p_bug); m.add(f"{name} {spec} {feedback or ''}", code)
            feedback = run_tests(code, tests)
            if feedback is None: ok += 1; break
    return ok

def multi_agent(rnd, p_bug, coder, m, max_rounds=3):
    # 规划者（1 次模型调用）→ 每个函数：编码者写 → 运行测试 → 审查者（模型）读代码和报错写修改意见 → 编码者改
    plan = [(n, TASKS[n][0]) for n in TASKS]; m.add("规划者：拆分需求" + json.dumps(TASKS, ensure_ascii=False), json.dumps(plan, ensure_ascii=False))
    ok = 0
    for name, spec in plan:
        feedback = None
        for _ in range(max_rounds):
            code = coder(name, spec, feedback, rnd, p_bug); m.add(f"编码者：{name} {spec} {feedback or ''}", code)
            err = run_tests(code, TASKS[name][1])
            review = "通过" if err is None else f"测试失败：{err}，请检查边界情况"
            m.add("审查者：阅读代码与测试结果\n" + code + str(err), review)  # 审查者也是一次模型调用
            if err is None: ok += 1; break
            feedback = review
    return ok

if __name__ == "__main__":
    use_llm = bool(os.getenv("OPENAI_API_KEY")); coder = coder_llm if use_llm else coder_mock
    trials = 3 if use_llm else 2000; res = {}
    for p_bug in [0.1, 0.3, 0.5]:
        for mode, fn in [("单Agent", single_agent), ("单Agent+自动测试", single_with_tests), ("多Agent", multi_agent)]:
            rnd = random.Random(42); m = Meter(); passed = sum(fn(rnd, p_bug, coder, m) for _ in range(trials))
            res[f"{mode}@{p_bug}"] = dict(pass_rate=passed / (3 * trials), calls=m.calls / trials, chars=m.chars / trials)
            print(f"出错概率 {p_bug:.1f}  {mode:12s}: 函数通过率 {passed/(3*trials):.1%}  每任务模型调用 {m.calls/trials:.1f} 次  上下文 {m.chars/trials:.0f} 字符")
    os.makedirs("outputs", exist_ok=True); json.dump(res, open("outputs/results.json", "w"), ensure_ascii=False, indent=1)
