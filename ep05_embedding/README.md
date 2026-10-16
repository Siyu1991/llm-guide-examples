# ep05 Embedding：关键词检索 vs 向量检索

对应专栏第 5 期。使用 fastembed（ONNX Runtime，无需 PyTorch/GPU）运行 BAAI/bge-small-zh-v1.5（约 95MB）。

```bash
pip install -r requirements.txt
python search_demo.py --offline   # 只跑 BM25，不下载模型
python search_demo.py             # 首次运行会从 Hugging Face 下载模型
```

国内网络可先设置 `export HF_ENDPOINT=https://hf-mirror.com`。峰值内存约 0.8GB（主要来自 30 万条随机向量的暴力检索测试）。

`data.py` 是自编的 40 件商品和 18 句口语查询，规模很小，结果只用来演示差异，不代表模型在真实业务上的表现。
