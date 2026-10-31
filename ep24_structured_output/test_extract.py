#!/usr/bin/env python3
"""离线单测：schema、剥离 JSON、重试收敛。"""
import json
import unittest

from pydantic import ValidationError

from extract import (
    MOCK_RESPONSES,
    MockModel,
    TicketExtract,
    extract_with_retry,
    strip_to_json,
)


class TestSchema(unittest.TestCase):
    def test_valid(self):
        obj = TicketExtract(
            category="billing",
            priority="high",
            customer_email="Alice@Ex.COM",
            summary="发票金额不对需要核对",
            order_id="ORD-99102",
        )
        self.assertEqual(obj.customer_email, "alice@ex.com")

    def test_bad_email(self):
        with self.assertRaises(ValidationError):
            TicketExtract(
                category="billing",
                priority="low",
                customer_email="not-an-email",
                summary="足够长的摘要文本",
            )

    def test_bad_order(self):
        with self.assertRaises(ValidationError):
            TicketExtract(
                category="other",
                priority="medium",
                customer_email="a@b.com",
                summary="足够长的摘要文本",
                order_id="123",
            )

    def test_bad_enum(self):
        with self.assertRaises(ValidationError):
            TicketExtract(
                category="Billing",
                priority="High",
                customer_email="a@b.com",
                summary="足够长的摘要文本",
            )


class TestStrip(unittest.TestCase):
    def test_strip(self):
        raw = '说明一下\n{"a":1,"b":"x"}\n完'
        self.assertEqual(json.loads(strip_to_json(raw)), {"a": 1, "b": "x"})


class TestRetry(unittest.TestCase):
    def test_converges(self):
        model = MockModel(MOCK_RESPONSES)
        result, attempts = extract_with_retry("dummy ticket", model)
        self.assertIsNotNone(result)
        self.assertEqual(result.order_id, "ORD-99102")
        self.assertEqual(result.priority.value, "high")
        self.assertGreaterEqual(len(attempts), 2)
        self.assertTrue(attempts[-1]["ok"])
        self.assertFalse(attempts[0]["ok"])

    def test_exhausts(self):
        model = MockModel(['not json'] * 3)
        result, attempts = extract_with_retry("x", model, max_retries=3)
        self.assertIsNone(result)
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
