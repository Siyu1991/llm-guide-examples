from agent import run_agent, MockLLM
import tools
def test_all():
    llm = MockLLM()
    assert "已发货" in run_agent("订单 A1001 到哪了", llm, verbose=False)[0]
    a, _ = run_agent("帮我订机票", llm, verbose=False); assert "确认" in a and not tools.BOOKINGS
    a, _ = run_agent("确认，帮我订机票", llm, verbose=False); assert "BK0001" in a and tools.BOOKINGS[0]["flight_no"] == "HO1253"
    assert "1817.5" in run_agent("A1001 A1002 A1003 汇总报表", llm, verbose=False)[0]
    assert "不存在" in run_agent("查 A9999", llm, verbose=False)[0]
    loop = lambda m, t: {"role": "assistant", "content": None, "tool_calls": [{"id": "x", "type": "function", "function": {"name": "query_order", "arguments": "{\"order_id\":\"A1001\"}"}}]}
    a, msgs = run_agent("死循环测试", loop, max_steps=3, verbose=False); assert "最大步数" in a
    assert "未知工具" in str(tools.call_tool("rm_rf", "{}")) and "JSON" in str(tools.call_tool("query_order", "{bad"))
if __name__ == "__main__":
    test_all(); print("ep09 tests passed")
