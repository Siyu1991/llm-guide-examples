import time, unittest
from server import start_server, STATE
from client import call_with_retry

class TestRetry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        STATE["fail_429_remaining"] = 2
        STATE["cut_stream"] = False
        cls.srv = start_server(8766)
        time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_recovers_after_429(self):
        STATE["fail_429_remaining"] = 2
        r = call_with_retry("http://127.0.0.1:8766/v1/stream", "abc", max_retries=5, base_delay=0.01)
        self.assertTrue(r["success"])
        self.assertGreaterEqual(len(r["attempts"]), 3)
        self.assertTrue(r["text"])

    def test_idem_key_stable(self):
        STATE["fail_429_remaining"] = 1
        r = call_with_retry("http://127.0.0.1:8766/v1/stream", "xyz", max_retries=4, base_delay=0.01)
        self.assertTrue(r["success"])
        self.assertTrue(r["idempotency_key"])

if __name__ == "__main__":
    unittest.main()
