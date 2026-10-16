"""估算 Agent 每一步发给模型的输入 token（工具定义每步都要重发）。需要 pip install tiktoken；没有则按字符数/1.5 粗估。"""
import json
from agent import run_agent, MockLLM, SYSTEM
from tools import SCHEMAS
try:
    import tiktoken; enc = tiktoken.get_encoding("o200k_base"); count = lambda s: len(enc.encode(s)); how = "tiktoken o200k_base"
except Exception:
    count = lambda s: int(len(s) / 1.5); how = "字符数/1.5 粗估"
tool_tok = count(json.dumps(SCHEMAS, ensure_ascii=False))
print(f"计数方式：{how}\n4 个工具定义合计 {tool_tok} token，系统提示 {count(SYSTEM)} token")
_, msgs = run_agent("确认，帮我订10月20日上海到北京最便宜的机票", MockLLM(), verbose=False)
total = 0
for i in range(1, len(msgs)):
    if msgs[i]["role"] == "assistant":
        ctx = tool_tok + count(json.dumps(msgs[:i], ensure_ascii=False)); total += ctx
        print(f"第 {sum(1 for m in msgs[:i+1] if m['role']=='assistant')} 次调用模型，输入约 {ctx} token")
print(f"订票任务 3 次模型调用，累计输入约 {total} token，是单次调用的 {total/(tool_tok + count(json.dumps(msgs[:2], ensure_ascii=False))):.1f} 倍")
