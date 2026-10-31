#!/usr/bin/env python3
"""本地 mock SSE 服务：可注入 429 / 中断 / 正常流。"""
from __future__ import annotations
import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"fail_429_remaining": 2, "cut_stream": False}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            req = {}
        idem = self.headers.get("Idempotency-Key", "")
        if STATE["fail_429_remaining"] > 0:
            STATE["fail_429_remaining"] -= 1
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "0")  # 测试里立刻重试
            self.send_header("x-ratelimit-remaining-requests", "0")
            self.end_headers()
            self.wfile.write(b'{"error":"rate_limit"}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        tokens = list("你好，这是一段流式输出。")
        if req.get("prompt"):
            tokens = list(str(req["prompt"])[:20]) + list("…ok")
        for i, ch in enumerate(tokens):
            if STATE["cut_stream"] and i == len(tokens) // 2:
                break  # 模拟中途断连
            payload = json.dumps({"delta": ch, "idem": idem}, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.01)
        if not STATE["cut_stream"]:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()


def start_server(port: int = 8765) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


if __name__ == "__main__":
    srv = start_server()
    print("listening 8765")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        srv.shutdown()
