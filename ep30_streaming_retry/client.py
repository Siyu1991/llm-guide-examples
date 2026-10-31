#!/usr/bin/env python3
"""带指数退避 + 抖动 + Idempotency-Key 的 SSE 客户端（对接本地 mock）。"""
from __future__ import annotations
import json, os, time, uuid, random, urllib.request, urllib.error

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def request_stream(url: str, prompt: str, idem: str, timeout: float = 5.0):
    data = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Content-Type": "application/json",
            "Idempotency-Key": idem,
            "Accept": "text/event-stream",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status == 200 and "text/event-stream" in resp.headers.get("Content-Type", ""):
            chunks = []
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                chunks.append(obj.get("delta", ""))
            return {"ok": True, "text": "".join(chunks), "status": 200}
        body = resp.read().decode("utf-8")
        return {"ok": False, "status": resp.status, "body": body}


def call_with_retry(url: str, prompt: str, max_retries: int = 5, base_delay: float = 0.05):
    idem = str(uuid.uuid4())
    attempts = []
    for i in range(max_retries):
        try:
            result = request_stream(url, prompt, idem)
            attempts.append({"attempt": i + 1, "ok": result.get("ok"), "status": result.get("status", 200)})
            if result.get("ok"):
                return {"success": True, "text": result["text"], "attempts": attempts, "idempotency_key": idem}
        except urllib.error.HTTPError as e:
            retry_after = e.headers.get("Retry-After")
            attempts.append({"attempt": i + 1, "ok": False, "status": e.code, "retry_after": retry_after})
            if e.code not in (429, 503):
                break
            # 指数退避 + 抖动；若有 Retry-After 则取较大者
            delay = base_delay * (2 ** i)
            delay *= 0.5 + random.random()  # jitter
            if retry_after is not None:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
            time.sleep(delay)
        except Exception as e:
            attempts.append({"attempt": i + 1, "ok": False, "error": str(e)})
            delay = base_delay * (2 ** i) * (0.5 + random.random())
            time.sleep(delay)
    return {"success": False, "text": None, "attempts": attempts, "idempotency_key": idem}


def main():
    from server import start_server, STATE
    import socket
    STATE["fail_429_remaining"] = 2
    STATE["cut_stream"] = False
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = start_server(port)
    time.sleep(0.1)
    result = call_with_retry(f"http://127.0.0.1:{port}/v1/stream", "hello-stream")
    path = os.path.join(OUT, "ep30_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("=== 流式 + 重试 ===")
    print(f"success={result['success']} attempts={len(result['attempts'])}")
    for a in result["attempts"]:
        print(" ", a)
    print("text:", result["text"])
    print("idem:", result["idempotency_key"])
    print("写入", path)
    srv.shutdown()


if __name__ == "__main__":
    main()
