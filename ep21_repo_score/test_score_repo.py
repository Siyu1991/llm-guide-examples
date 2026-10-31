#!/usr/bin/env python3
import json, os, unittest
from score_repo import score_readme, score_ci, FIX

class TestScore(unittest.TestCase):
    def test_readme_fixture(self):
        text = open(os.path.join(FIX, "README.md"), encoding="utf-8").read()
        r = score_readme(text)
        self.assertTrue(r["checks"]["mentions_docker"])
        self.assertGreaterEqual(r["score"], 40)

    def test_ci_single_os(self):
        ci = json.load(open(os.path.join(FIX, "ci_matrix.json"), encoding="utf-8"))
        c = score_ci(ci)
        self.assertFalse(c["checks"]["multi_os_matrix"])
        self.assertLess(c["score"], 40)

if __name__ == "__main__":
    unittest.main()
