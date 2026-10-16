"""Agent 上线必备的护栏：权限白名单、人工确认、预算熔断、重试+幂等、步数上限。纯标准库。"""
import time, uuid

class BudgetExceeded(Exception): pass
class PermissionDenied(Exception): pass

class Guard:
    def __init__(self, allow_read, allow_write, budget_usd, price_in=2e-6, price_out=1e-5, max_steps=10):
        self.allow_read, self.allow_write = set(allow_read), set(allow_write)
        self.budget, self.spent, self.steps, self.max_steps = budget_usd, 0.0, 0, max_steps
        self.price_in, self.price_out = price_in, price_out
        self.done_keys = {}  # 幂等：同一个 key 的写操作只执行一次
        self.audit = []
    def charge(self, tok_in, tok_out):
        self.steps += 1; self.spent += tok_in * self.price_in + tok_out * self.price_out
        if self.spent > self.budget: raise BudgetExceeded(f"已花 ${self.spent:.4f}，超过预算 ${self.budget}")
        if self.steps > self.max_steps: raise BudgetExceeded(f"超过 {self.max_steps} 步")
    def call(self, name, fn, args, confirm=None, idem_key=None, retries=2, backoff=0.01):
        if name not in self.allow_read | self.allow_write: raise PermissionDenied(f"{name} 不在白名单")
        if name in self.allow_write:
            if confirm is None or not confirm(name, args): self.audit.append(("拒绝", name, args)); return {"status": "需要人工确认"}
            idem_key = idem_key or str(uuid.uuid4())
            if idem_key in self.done_keys: return self.done_keys[idem_key]
        for i in range(retries + 1):
            try:
                r = fn(**args); self.audit.append(("成功", name, args))
                if name in self.allow_write: self.done_keys[idem_key] = r
                return r
            except TimeoutError:
                if i == retries: self.audit.append(("失败", name, args)); return {"status": "失败，已转人工"}
                time.sleep(backoff * 2 ** i)  # 指数退避
