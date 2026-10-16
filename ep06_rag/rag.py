"""第 6 期配套：一个能跑的最小 RAG。
流程：读 Markdown → 按标题和长度切块 → bge-small-zh-v1.5 向量化 → 余弦检索 top-k → 拼提示 → 生成。
生成：设置了 OPENAI_API_KEY 时调用 OpenAI 兼容接口（OPENAI_BASE_URL、RAG_MODEL 可改）；
     否则用离线"抽取式"回答：从检索到的片段里挑与问题最相似的句子原样返回，并附出处。
用法：
  python rag.py ask "去上海出差酒店能报多少"
  python rag.py eval            # 用 questions.json 评估检索命中率
  python rag.py ask ... --kb 你的文档目录
"""
import argparse, glob, json, os, re, time, resource, urllib.request
import numpy as np

def load_docs(kb):
    docs = []
    for p in sorted(glob.glob(os.path.join(kb, "**", "*.md"), recursive=True)):
        docs.append((os.path.basename(p), open(p, encoding="utf-8").read()))
    return docs

def chunk_markdown(name, text, size=300, overlap=50):
    """先按二级/三级标题分节，节内超长再按句子打包到 size 字左右，相邻块重叠 overlap 字。
    每块前面带上文档标题和小节标题，检索时才知道"这是 2026 版报销制度的住宿标准"。"""
    title = (re.findall(r"^# (.+)$", text, re.M) or [name])[0]
    parts = re.split(r"^(#{2,3} .+)$", text, flags=re.M)
    sections = [("", parts[0])] + [(parts[i].lstrip("# ").strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    out = []
    for head, body in sections:
        body = re.sub(r"^# .+$", "", body, flags=re.M).strip()
        if not body:
            continue
        pieces, buf = [], ""
        for s in (x for x in re.split(r"(?<=[。；！？\n])", body) if x):
            if len(buf) + len(s) > size and buf:
                pieces.append(buf); buf = buf[-overlap:] if overlap else ""
            buf += s
        if buf.strip():
            pieces.append(buf)
        out += [{"doc": name, "title": title, "section": head, "text": f"【{title}｜{head}】" + p.strip()} for p in pieces]
    return out

class Embedder:
    def __init__(self, name="BAAI/bge-small-zh-v1.5"):
        from fastembed import TextEmbedding
        self.m = TextEmbedding(name)
    def docs(self, texts):
        X = np.array(list(self.m.embed(texts))); return X / np.linalg.norm(X, axis=1, keepdims=True)
    def query(self, q):
        x = np.array(list(self.m.query_embed([q])))[0]; return x / np.linalg.norm(x)

class Index:
    def __init__(self, kb, size=300, overlap=50, emb=None):
        self.chunks = [c for n, t in load_docs(kb) for c in chunk_markdown(n, t, size, overlap)]
        self.emb = emb or Embedder()
        t = time.perf_counter(); self.X = self.emb.docs([c["text"] for c in self.chunks]); self.build_s = time.perf_counter() - t
    def search(self, q, k=4):
        s = self.X @ self.emb.query(q)
        idx = np.argsort(-s)[:k]
        return [dict(self.chunks[i], score=float(s[i])) for i in idx]

PROMPT = """你是公司内部知识库助手。只根据下面的资料回答问题，资料里没有的内容就回答"资料中没有找到"。
回答要简短，并在句末用 [编号] 标出依据的资料。如果资料之间有冲突，以标注为现行、版本更新的为准。

资料：
{ctx}

问题：{q}"""

def build_prompt(q, hits):
    ctx = "\n".join(f"[{i+1}] {h['text']}" for i, h in enumerate(hits))
    return PROMPT.format(ctx=ctx, q=q)

def generate_api(prompt):
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    body = {"model": os.environ.get("RAG_MODEL", "gpt-4o-mini"), "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
          headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("usage")

def generate_offline(q, hits, emb):
    """离线兜底：不调用大模型，挑出与问题最相似的一句原文。能演示"有依据地回答"，但不会推理和改写。"""
    cands = []
    for i, h in enumerate(hits):
        body = h["text"].split("】", 1)[-1]
        for s in re.split(r"(?<=[。；\n])", body):
            if len(s.strip()) > 4: cands.append((s.strip(), i + 1))
    if not cands: return "资料中没有找到", None
    S = emb.docs([c[0] for c in cands]) @ emb.query(q)
    j = int(np.argmax(S))
    return f"{cands[j][0]} [{cands[j][1]}]", {"mode": "offline-extractive"}

def check_numbers(answer, hits):
    """后处理核对：回答中的每个数字，必须出现在它所引用的 [编号] 片段里；没标编号时对照全部片段。返回找不到依据的数字。"""
    cited = [int(i) for i in re.findall(r"\[(\d+)\]", answer) if 0 < int(i) <= len(hits)]
    pool = "".join(hits[i - 1]["text"] for i in cited) if cited else "".join(h["text"] for h in hits)
    body = re.sub(r"\[\d+\]", "", answer)
    return [n for n in re.findall(r"\d+(?:\.\d+)?", body) if n not in pool]

def evaluate(index, qs, k=4):
    hit = 0; rows = []
    for item in qs:
        hits = index.search(item["q"], k)
        ok = any(item["a"] in h["text"] for h in hits)
        hit += ok; rows.append((ok, item["q"], hits[0]["doc"]))
    return hit / len(qs), rows

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ask", "eval"]); ap.add_argument("question", nargs="?")
    ap.add_argument("--kb", default=os.path.join(here, "kb")); ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--size", type=int, default=300); ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    idx = Index(a.kb, size=a.size)
    print(f"知识库：{len(idx.chunks)} 个片段，向量化耗时 {idx.build_s:.2f}s")
    if a.cmd == "ask":
        hits = idx.search(a.question, a.k)
        for i, h in enumerate(hits): print(f"  [{i+1}] {h['score']:.3f} {h['text'][:60]}…")
        prompt = build_prompt(a.question, hits)
        if os.environ.get("OPENAI_API_KEY") and not a.offline:
            ans, usage = generate_api(prompt)
        else:
            ans, usage = generate_offline(a.question, hits, idx.emb)
        print("\n答：", ans, "\n用量：", usage, f"\n提示长度：{len(prompt)} 字符")
        bad = check_numbers(ans, hits)
        print("数字核对：", "通过" if not bad else f"以下数字在所引片段中找不到：{bad}")
    else:
        qs = json.load(open(os.path.join(here, "questions.json"), encoding="utf-8"))
        rate, rows = evaluate(idx, qs, a.k)
        for ok, q, d in rows: print(("✓ " if ok else "✗ ") + q, "→", d)
        print(f"hit@{a.k} = {rate:.3f}")
        assert rate >= 0.7, "检索命中率过低，检查模型是否下载完整"
    print(f"峰值内存 RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.0f} MB")
    print("OK")

if __name__ == "__main__":
    main()
