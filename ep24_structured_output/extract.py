#!/usr/bin/env python3
"""结构化输出：Pydantic schema + 校验 + 重试（离线 mock，不需要 API key）。

模拟「工单分类 / 表单抽取」场景：模型返回可能缺字段、枚举拼错、或夹带推理文本。
策略：严格 schema 校验 → 失败则把错误反馈给 mock 模型再试，最多 N 次。
"""
from __future__ import annotations

import json
import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class Category(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class TicketExtract(BaseModel):
    category: Category
    priority: Priority
    customer_email: str = Field(..., min_length=5)
    summary: str = Field(..., min_length=8, max_length=200)
    order_id: Optional[str] = None
    needs_human: bool = False

    @field_validator("customer_email")
    @classmethod
    def email_has_at(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("customer_email 必须是合法邮箱形态")
        return v.strip().lower()

    @field_validator("order_id")
    @classmethod
    def order_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not (v.startswith("ORD-") and len(v) >= 8):
            raise ValueError("order_id 形如 ORD-XXXXXX")
        return v


# 故意设计的「坏→好」响应序列，模拟模型被纠错后逐步收敛
MOCK_RESPONSES = [
    # 1) 非 JSON + 枚举错误
    '先分析一下……我觉得这是账单问题。\n{"category":"Billing","priority":"High","customer_email":"alice@ex.com","summary":"发票金额不对"}',
    # 2) JSON 合法但枚举仍错、邮箱缺域名点
    '{"category":"billing","priority":"High","customer_email":"alice@ex","summary":"发票金额不对","order_id":"123"}',
    # 3) 仍有 order_id 格式问题
    '{"category":"billing","priority":"high","customer_email":"alice@ex.com","summary":"发票金额不对，需要核对","order_id":"12345","needs_human":false}',
    # 4) 通过
    '{"category":"billing","priority":"high","customer_email":"alice@ex.com","summary":"发票金额不对，需要核对","order_id":"ORD-99102","needs_human":false}',
]


class MockModel:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls = 0
        self.history: list[dict] = []

    def complete(self, prompt: str) -> str:
        if self.calls >= len(self.responses):
            raise RuntimeError("mock 响应耗尽")
        text = self.responses[self.calls]
        self.calls += 1
        self.history.append({"prompt_tail": prompt[-200:], "response": text})
        return text


def strip_to_json(text: str) -> str:
    """从可能夹带说明的文本里抽出第一个 JSON 对象。"""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("响应中找不到 JSON 对象")
    return text[start : end + 1]


def extract_with_retry(
    raw_ticket: str,
    model: MockModel,
    max_retries: int = 4,
) -> tuple[Optional[TicketExtract], list[dict]]:
    schema_hint = json.dumps(TicketExtract.model_json_schema(), ensure_ascii=False)
    attempts: list[dict] = []
    feedback = ""
    for i in range(max_retries):
        prompt = (
            "从工单文本抽取结构化字段，只输出 JSON。\n"
            f"Schema: {schema_hint}\n"
            f"工单: {raw_ticket}\n"
            f"{feedback}"
        )
        raw = model.complete(prompt)
        record: dict = {"attempt": i + 1, "raw": raw, "ok": False, "error": None}
        try:
            blob = strip_to_json(raw)
            data = json.loads(blob)
            obj = TicketExtract.model_validate(data)
            record["ok"] = True
            record["parsed"] = obj.model_dump()
            attempts.append(record)
            return obj, attempts
        except (ValueError, json.JSONDecodeError, ValidationError) as e:
            record["error"] = str(e)
            attempts.append(record)
            feedback = f"上次输出校验失败，请修正后只输出 JSON。错误: {e}\n"
    return None, attempts


def main():
    ticket = (
        "来自 alice@ex.com：上月订单 ORD-99102 发票金额多扣了 30 元，"
        "请帮忙核对账单。优先级希望高一点。"
    )
    model = MockModel(MOCK_RESPONSES)
    result, attempts = extract_with_retry(ticket, model)
    out = {
        "ticket": ticket,
        "success": result is not None,
        "result": result.model_dump() if result else None,
        "attempts": attempts,
        "model_calls": model.calls,
    }
    path = os.path.join(OUT, "ep24_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("=== 结构化抽取 + 校验重试 ===")
    for a in attempts:
        status = "OK" if a["ok"] else "FAIL"
        print(f"attempt {a['attempt']}: {status}")
        if a["error"]:
            print(f"  error: {a['error'][:120]}")
    if result:
        print("最终结果:")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    print(f"写入 {path}")


if __name__ == "__main__":
    main()
