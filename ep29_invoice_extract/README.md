# ep29 票据识别（离线）

合成发票 PNG + mock 视觉输出 + Pydantic 校验。并用 OpenAI 文档中的 32px patch 公式估算图像 patch 数（未乘模型倍率）。

```bash
pip install -r requirements.txt
python3 extract.py
python3 -m unittest test_extract -v
```
