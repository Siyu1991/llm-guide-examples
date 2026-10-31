import unittest
from agent import run_case, CASES, authorize, ToolCall

class TestAgent(unittest.TestCase):
    def test_shell_denied_for_agent(self):
        r = run_case(CASES[0])
        self.assertFalse(r["allowed"])
        self.assertIn("denied_risk", r["reason"])

    def test_indirect_refund_still_needs_acl(self):
        # agent 角色允许 write，故解析后会 allowed——说明「仅靠提示无法防间接注入」，需业务确认闸
        r = run_case(CASES[1])
        self.assertTrue(r["allowed"])  # ACL 放行 write；生产要再加 human confirm
        self.assertEqual(r["parsed"]["name"], "refund_order")

    def test_legitimate(self):
        r = run_case(CASES[2])
        self.assertTrue(r["allowed"])

    def test_viewer_denied(self):
        r = run_case(CASES[3])
        self.assertFalse(r["allowed"])

    def test_authorize_unknown(self):
        ok, reason = authorize("admin", ToolCall("nope", {}))
        self.assertFalse(ok)

if __name__ == "__main__":
    unittest.main()
