# ep30 流式输出与重试

本地 mock SSE 服务（可先返回 429 + `Retry-After`）+ 客户端指数退避与抖动、固定 `Idempotency-Key`。

```bash
python3 client.py
python3 -m unittest test_client -v
```
