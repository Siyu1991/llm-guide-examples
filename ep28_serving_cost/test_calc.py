import unittest
from calc import kv_bytes_per_token, estimate_kv_gib, ModelCfg, breakeven

class TestCalc(unittest.TestCase):
    def test_bytes(self):
        # 32 layers, 8 kv heads, 128 dim, fp16 → 32*2*8*128*2 = 131072
        self.assertEqual(kv_bytes_per_token(32, 8, 128), 131072)

    def test_gib_positive(self):
        cfg = ModelCfg("t", 32, 8, 128)
        self.assertGreater(estimate_kv_gib(cfg, 8, 4096), 1.0)

    def test_breakeven_monotonic(self):
        a = breakeven("RTX_4090", 10_000)
        b = breakeven("RTX_4090", 500_000)
        self.assertFalse(a["self_host_cheaper"])
        self.assertTrue(b["self_host_cheaper"])

if __name__ == "__main__":
    unittest.main()
