#!/usr/bin/env python3
import unittest
from claim_check import split_comparison, independent_evidence_count, Source, cost_per_success

class TestClaim(unittest.TestCase):
    def test_split(self):
        parts = split_comparison("更快、更便宜，也更准确")
        self.assertGreaterEqual(len(parts), 3)

    def test_roots(self):
        src = [
            Source("u1", "rootA", "2026-01-01", ["c2"]),
            Source("u2", "rootA", "2026-01-02", ["c2"]),
            Source("u3", "rootB", "2026-01-03", ["c2"]),
        ]
        ev = independent_evidence_count(src, "c2")
        self.assertEqual(ev["link_count"], 3)
        self.assertEqual(ev["independent_roots"], 2)

    def test_cost(self):
        self.assertAlmostEqual(cost_per_success(0.005, 0.5), 0.01)
        self.assertAlmostEqual(cost_per_success(0.01, 0.9), 0.01 / 0.9)

if __name__ == "__main__":
    unittest.main()
