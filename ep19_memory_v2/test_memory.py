#!/usr/bin/env python3
import os, unittest, tempfile
from memory_store import MemoryStore, jaccard

class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        self.tmp.close()
        os.unlink(self.tmp.name)
        self.store = MemoryStore(self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_jaccard_identical(self):
        self.assertEqual(jaccard("不吃香菜", "不吃香菜"), 1.0)

    def test_user_isolation(self):
        self.store.add("a", "喜欢咖啡", infer_text="喜欢咖啡")
        self.store.add("b", "喜欢茶", infer_text="喜欢茶")
        hits = self.store.search("a", "喝什么")
        self.assertTrue(all(m.user_id == "a" for m, _ in hits))

    def test_other_subject_not_self(self):
        self.store.add("u", "朋友不吃香菜", infer_text="朋友不吃香菜", subject="other:friend")
        hits = self.store.search("u", "香菜")
        self.assertEqual(hits[0][0].subject, "other:friend")

    def test_conflict_and_forget(self):
        self.store.add("u", "不吃香菜", infer_text="用户不吃香菜")
        self.store.update_conflict("u", "不吃香菜", "用户可以吃香菜", "可以吃了")
        active = [m for m in self.store.items if m.active and m.user_id == "u"]
        self.assertEqual(len(active), 1)
        self.assertIn("可以吃", active[0].text)
        self.store.forget("u", "可以吃")
        self.assertEqual([m for m in self.store.items if m.active], [])

if __name__ == "__main__":
    unittest.main()
