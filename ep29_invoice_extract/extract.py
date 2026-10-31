#!/usr/bin/env python3
"""票据字段抽取：合成发票图 + mock 视觉模型输出 + schema 校验（离线）。"""
from __future__ import annotations
import json, os
from typing import Optional
from pydantic import BaseModel, Field, ValidationError, field_validator

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)
SAMPLE = os.path.join(os.path.dirname(__file__), "sample_invoice.png")


class InvoiceFields(BaseModel):
    invoice_no: str = Field(..., min_length=5)
    seller: str
    buyer: str
    amount_cny: float = Field(..., gt=0)
    currency: str = "CNY"
    invoice_date: str  # YYYY-MM-DD

    @field_validator("invoice_no")
    @classmethod
    def inv_prefix(cls, v: str) -> str:
        if not v.upper().startswith("INV-"):
            raise ValueError("invoice_no 应以 INV- 开头")
        return v.upper()

    @field_validator("invoice_date")
    @classmethod
    def date_shape(cls, v: str) -> str:
        parts = v.split("-")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("invoice_date 应为 YYYY-MM-DD")
        return v


# mock「视觉模型」：好/坏输出
MOCKS = [
    '{"invoice_no":"inv-2026-1001","seller":"Example Supplies Ltd.","buyer":"Acme Retail Co.","amount_cny":128.50,"invoice_date":"2026-10-30"}',
    '{"invoice_no":"2026-1001","seller":"Example","buyer":"Acme","amount_cny":-1,"invoice_date":"10/30/2026"}',
]


def image_patch_tokens(width: int, height: int, patch: int = 32) -> int:
    """OpenAI 文档：32px patch，patch_count = ceil(w/32)*ceil(h/32)（未乘模型倍率）。"""
    import math
    return math.ceil(width / patch) * math.ceil(height / patch)


def try_parse(raw: str) -> tuple[Optional[InvoiceFields], Optional[str]]:
    try:
        data = json.loads(raw)
        return InvoiceFields.model_validate(data), None
    except (json.JSONDecodeError, ValidationError) as e:
        return None, str(e)


def main():
    from PIL import Image
    im = Image.open(SAMPLE)
    patches = image_patch_tokens(im.width, im.height)
    # 文档：再乘模型倍率；此处只报告 patch 数，倍率留给读者按当日表查
    attempts = []
    final = None
    for i, raw in enumerate(MOCKS, 1):
        obj, err = try_parse(raw)
        attempts.append({"attempt": i, "raw": raw, "ok": obj is not None, "error": err,
                         "parsed": obj.model_dump() if obj else None})
        if obj:
            final = obj
            break
    out = {
        "sample_image": SAMPLE,
        "image_size": [im.width, im.height],
        "patch_32_count": patches,
        "note": "billable tokens = patch_count * model_multiplier (see OpenAI Images and vision docs)",
        "attempts": attempts,
        "success": final is not None,
        "result": final.model_dump() if final else None,
    }
    path = os.path.join(OUT, "ep29_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("=== 票据抽取 ===")
    print(f"image {im.width}x{im.height} patch32={patches}")
    for a in attempts:
        print(f"attempt {a['attempt']}: {'OK' if a['ok'] else 'FAIL'}")
        if a["error"]:
            print(" ", a["error"][:100])
    if final:
        print(json.dumps(final.model_dump(), ensure_ascii=False, indent=2))
    print("写入", path)


if __name__ == "__main__":
    main()
