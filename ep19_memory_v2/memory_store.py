#!/usr/bin/env python3
"""最小应用层记忆：写入抽取、检索、冲突更新、遗忘。

不依赖外部向量库与 API。相似度用字符 3-gram Jaccard（离线可复现）。
字段设计参考 Mem0 文档中的 user/session 作用域与 additive add 行为说明，
本实现是教学用本地存储，不是 Mem0 SDK 封装。
"""
from __future__ import annotations
import json, os, re, time, uuid
from dataclasses import dataclass, asdict, field
from typing import Optional

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)

STORE_PATH = os.path.join(OUT, "memories.json")


def grams(text: str, n: int = 2) -> set[str]:
    t = re.sub(r"\s+", "", text.lower())
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: str, b: str) -> float:
    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 1.0 if a and a in b else 0.0
    score = len(ga & gb) / len(ga | gb)
    if a and a in b:
        score = max(score, 0.5)
    return score


@dataclass
class Memory:
    id: str
    user_id: str
    text: str
    subject: str  # self | other:<name> | unknown
    scope: str  # permanent | until:<ISO date> | session
    source_quote: str
    created_at: float
    updated_at: float
    active: bool = True
    superseded_by: Optional[str] = None


class MemoryStore:
    def __init__(self, path: str = STORE_PATH):
        self.path = path
        self.items: list[Memory] = []
        if os.path.exists(path):
            self._load()

    def _load(self):
        raw = json.load(open(self.path, encoding="utf-8"))
        self.items = [Memory(**m) for m in raw]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.items], f, ensure_ascii=False, indent=2)

    def add(
        self,
        user_id: str,
        utterance: str,
        *,
        subject: str = "self",
        scope: str = "permanent",
        source_quote: str | None = None,
        infer_text: str | None = None,
    ) -> Memory:
        text = infer_text if infer_text is not None else utterance.strip()
        now = time.time()
        m = Memory(
            id=str(uuid.uuid4())[:8],
            user_id=user_id,
            text=text,
            subject=subject,
            scope=scope,
            source_quote=source_quote or utterance,
            created_at=now,
            updated_at=now,
        )
        self.items.append(m)
        self.save()
        return m

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[tuple[Memory, float]]:
        scored = []
        for m in self.items:
            if not m.active or m.user_id != user_id:
                continue
            if m.subject != "self" and not m.subject.startswith("other"):
                pass
            scored.append((m, jaccard(query, m.text)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(m, s) for m, s in scored[:top_k] if s > 0]

    def update_conflict(self, user_id: str, old_substr: str, new_text: str, quote: str) -> dict:
        """若存在含 old_substr 的 active 记忆，标记失效并写入新记忆。"""
        touched = []
        new_id = None
        for m in self.items:
            if m.active and m.user_id == user_id and m.subject == "self" and old_substr in m.text:
                m.active = False
                nm = self.add(user_id, quote, infer_text=new_text, source_quote=quote)
                m.superseded_by = nm.id
                m.updated_at = time.time()
                new_id = nm.id
                touched.append(m.id)
        self.save()
        return {"deactivated": touched, "new_id": new_id}

    def forget(self, user_id: str, substr: str) -> list[str]:
        ids = []
        for m in self.items:
            if m.active and m.user_id == user_id and substr in m.text:
                m.active = False
                m.updated_at = time.time()
                ids.append(m.id)
        self.save()
        return ids


def demo() -> dict:
    path = os.path.join(OUT, "demo_memories.json")
    if os.path.exists(path):
        os.remove(path)
    store = MemoryStore(path)
    log = []

    m1 = store.add("u1", "我不吃香菜", infer_text="用户不吃香菜", subject="self")
    log.append({"step": "add_self", "id": m1.id, "text": m1.text})

    m2 = store.add(
        "u1",
        "朋友小李不吃香菜",
        infer_text="小李不吃香菜",
        subject="other:小李",
    )
    log.append({"step": "add_other", "id": m2.id, "subject": m2.subject})

    hits = store.search("u1", "推荐晚餐不要香菜")
    log.append({
        "step": "search_dinner",
        "hits": [{"id": m.id, "text": m.text, "subject": m.subject, "score": round(s, 3)} for m, s in hits],
    })

    # 应用层过滤：推荐给本人时只用 subject=self
    self_hits = [(m, s) for m, s in hits if m.subject == "self"]
    log.append({"step": "filter_self_only", "count": len(self_hits), "texts": [m.text for m, _ in self_hits]})

    upd = store.update_conflict("u1", "不吃香菜", "用户现在可以吃香菜", "现在我可以吃香菜了")
    log.append({"step": "conflict_update", **upd})

    hits2 = store.search("u1", "香菜晚餐")
    log.append({
        "step": "search_after_update",
        "active_hits": [{"text": m.text, "active": m.active, "score": round(s, 3)} for m, s in hits2],
    })

    forgotten = store.forget("u1", "可以吃香菜")
    log.append({"step": "forget", "ids": forgotten})
    hits3 = store.search("u1", "香菜")
    log.append({"step": "search_after_forget", "count": len(hits3)})

    out = {"log": log, "store_path": path}
    with open(os.path.join(OUT, "ep19_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


if __name__ == "__main__":
    r = demo()
    print(json.dumps(r["log"], ensure_ascii=False, indent=2))
