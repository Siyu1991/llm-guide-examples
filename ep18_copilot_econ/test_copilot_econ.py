#!/usr/bin/env python3
import json, os, unittest
from copilot_econ import (
    PLANS, CREDIT_USD, usd_for_turn, Turn, session_quick_chat,
    session_agent_long, credits_for_usd, sessions_until_empty,
)

class TestEcon(unittest.TestCase):
    def test_credit_rate(self):
        self.assertEqual(CREDIT_USD, 0.01)
        self.assertEqual(credits_for_usd(1.0), 100.0)

    def test_pro_allowance(self):
        self.assertEqual(PLANS["Pro"]["credits"], 1500)
        self.assertEqual(PLANS["Pro"]["price_usd"], 10)

    def test_turn_math(self):
        # 1M in @ $2 = $2
        u = usd_for_turn("gpt-6.1-sol", Turn(1_000_000, 0, 0, 0))
        self.assertAlmostEqual(u, 2.0, places=9)

    def test_quick_positive(self):
        s = session_quick_chat("gpt-6.1-sol")
        self.assertGreater(s["credits"], 0)
        cap = sessions_until_empty("Pro", s)
        self.assertGreater(cap["full_sessions"], 10)

    def test_agent_costs_more(self):
        q = session_quick_chat("gpt-6.1-sol")
        a = session_agent_long("gpt-6.1-sol")
        self.assertGreater(a["credits"], q["credits"])

if __name__ == "__main__":
    unittest.main()
