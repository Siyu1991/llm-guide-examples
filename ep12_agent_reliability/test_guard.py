from guard import Guard, BudgetExceeded, PermissionDenied
from simulate import run
def test():
    g = Guard(["query"], ["refund"], budget_usd=0.01)
    try: g.call("delete_db", lambda: 1, {}); assert False
    except PermissionDenied: pass
    assert g.call("refund", lambda amount: "ok", {"amount": 5}, confirm=lambda n, a: False)["status"] == "需要人工确认"
    n = {"c": 0}
    def refund(amount): n["c"] += 1; return {"refunded": amount}
    yes = lambda nm, a: a["amount"] <= 100
    g.call("refund", refund, {"amount": 5}, confirm=yes, idem_key="k1"); g.call("refund", refund, {"amount": 5}, confirm=yes, idem_key="k1")
    assert n["c"] == 1  # 幂等：重复请求只退一次款
    flaky = {"c": 0}
    def q():
        flaky["c"] += 1
        if flaky["c"] < 3: raise TimeoutError
        return "ok"
    assert g.call("query", q, {}) == "ok"
    try:
        for _ in range(100): g.charge(1500, 200)
        assert False
    except BudgetExceeded: pass
    assert abs(run(10, 0.95, 0, False, 20000)["success"] - 0.95 ** 10) < 0.02
if __name__ == "__main__": test(); print("ep12 tests passed")
