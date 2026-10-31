import unittest
from extract import try_parse, image_patch_tokens, InvoiceFields

class TestInv(unittest.TestCase):
    def test_ok(self):
        obj, err = try_parse('{"invoice_no":"INV-1","seller":"S","buyer":"B","amount_cny":1,"invoice_date":"2026-01-02"}')
        self.assertIsNotNone(obj)
        self.assertEqual(obj.invoice_no, "INV-1")

    def test_bad_amount(self):
        obj, err = try_parse('{"invoice_no":"INV-1","seller":"S","buyer":"B","amount_cny":-1,"invoice_date":"2026-01-02"}')
        self.assertIsNone(obj)

    def test_patches(self):
        # 640x400 → ceil(20)*ceil(12.5)=20*13=260
        self.assertEqual(image_patch_tokens(640, 400), 260)

if __name__ == "__main__":
    unittest.main()
