import unittest
from harness import MOCK_MODELS, eval_model, regression, score_case, GOLDEN

class TestEval(unittest.TestCase):
    def test_v1_all_pass(self):
        r = eval_model("model_v1", MOCK_MODELS["model_v1"])
        self.assertEqual(r["pass_rate"], 1.0)

    def test_v2_regresses(self):
        a = eval_model("model_v1", MOCK_MODELS["model_v1"])
        b = eval_model("model_v2", MOCK_MODELS["model_v2"])
        g = regression(a, b)
        self.assertTrue(g["gate_failed"])
        self.assertLess(b["pass_rate"], a["pass_rate"])

    def test_score_case(self):
        self.assertTrue(score_case("退货窗口是7天", GOLDEN[0]))
        self.assertFalse(score_case("不清楚", GOLDEN[0]))

if __name__ == "__main__":
    unittest.main()
