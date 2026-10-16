"""最小 Agent：模型决定调用哪个工具 → 程序执行 → 结果回填 → 循环，直到模型给出最终回答。
有 OPENAI_API_KEY 时调用 OpenAI 兼容接口（chat/completions + tools）；否则用 MockLLM 离线演示同样的消息格式。
"""
import json, os, re, sys, urllib.request
from tools import SCHEMAS, call_tool

class MockLLM:
    """离线替身：用规则模拟"模型输出 tool_calls"。只为演示循环和消息格式，不代表真实模型能力。"""
    def __call__(self, messages, tools):
        user = next(m["content"] for m in messages if m["role"] == "user")
        done = [m for m in messages if m["role"] == "tool"]
        last = json.loads(done[-1]["content"]) if done else None
        def tc(name, args): return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"call_{len(done)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}
        ids = re.findall(r"A\d{4}", user)
        if "报表" in user or "汇总" in user:
            if not done: return tc("sales_report", {"order_ids": ids})
            return {"role": "assistant", "content": f"共 {last['count']} 单，合计 {last['total']} 元。"}
        if ids:
            if not done: return tc("query_order", {"order_id": ids[0]})
            if "error" in last: return {"role": "assistant", "content": last["error"] + "，请核对订单号。"}
            return {"role": "assistant", "content": f"订单 {last['order_id']} {last['status']}，承运 {last['carrier'] or '暂无'}，预计 {last['eta'] or '已完成'}。"}
        if "机票" in user or "航班" in user:
            if not done: return tc("search_flights", {"origin": "上海", "dest": "北京", "date": "2026-10-20"})
            if len(done) == 1:
                cheapest = min(last, key=lambda f: f["price"])
                return tc("book_flight", {"flight_no": cheapest["no"], "passenger": "张三", "confirmed": "确认" in user})
            if last.get("need_confirmation"): return {"role": "assistant", "content": last["msg"] + "（回复'确认'后下单）"}
            return {"role": "assistant", "content": f"已预订，单号 {last['booking_id']}。"}
        return {"role": "assistant", "content": "抱歉，这个问题我处理不了，已转人工。"}

def openai_llm(messages, tools):
    body = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages": messages, "tools": tools, "temperature": 0}
    req = urllib.request.Request(os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1") + "/chat/completions",
        data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    msg = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]
    return {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}

SYSTEM = "你是网店客服助手。只能通过工具获取订单和航班信息，不要编造。涉及付款或下单的操作必须先让用户确认。"

def run_agent(user_text, llm=None, max_steps=6, verbose=True):
    llm = llm or (openai_llm if os.getenv("OPENAI_API_KEY") else MockLLM())
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    for step in range(max_steps):
        msg = llm(messages, SCHEMAS); messages.append(msg)
        if not msg.get("tool_calls"):
            if verbose: print(f"  [第{step+1}步] 最终回答：{msg['content']}")
            return msg["content"], messages
        for c in msg["tool_calls"]:
            res = call_tool(c["function"]["name"], c["function"]["arguments"])
            if verbose: print(f"  [第{step+1}步] 调用 {c['function']['name']}({c['function']['arguments']}) → {json.dumps(res, ensure_ascii=False)[:120]}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(res, ensure_ascii=False)})
    return "超过最大步数，已转人工。", messages  # 兜底：防止死循环烧钱

if __name__ == "__main__":
    tasks = sys.argv[1:] or ["我的订单 A1001 到哪了？", "帮我订10月20日上海到北京最便宜的机票", "确认，帮我订10月20日上海到北京最便宜的机票",
                             "把 A1001 A1002 A1003 做个销售汇总报表", "查一下 A9999", "帮我写首诗"]
    print("模型：", "OpenAI 兼容接口" if os.getenv("OPENAI_API_KEY") else "MockLLM（离线）")
    for t in tasks:
        print("用户：", t); run_agent(t)
