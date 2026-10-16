# ep05 Embedding：关键词检索 vs 向量检索

对应专栏第 5 期。使用 fastembed（ONNX Runtime，无需 PyTorch/GPU）运行 BAAI/bge-small-zh-v1.5（约 95MB）。

```bash
pip install -r requirements.txt
python search_demo.py --offline   # 只跑 BM25，不下载模型
python search_demo.py             # 首次运行会从 Hugging Face 下载模型
```

国内网络可先设置 `export HF_ENDPOINT=https://hf-mirror.com`。峰值内存约 0.8GB（主要来自 30 万条随机向量的暴力检索测试）。

`data.py` 是自编的 40 件商品和 18 句口语查询，规模很小，结果只用来演示差异，不代表模型在真实业务上的表现。

## 补充：三种 ANN 索引对比

```bash
pip install faiss-cpu
python ann_bench.py            # 20 万条合成向量，单线程约 2 分钟，峰值内存约 1GB
python ann_bench.py --n 50000  # 机器较弱时调小数据量
```

对比 Flat 暴力检索、HNSW（不同 efSearch）、IVF-PQ（单独使用 / 加原始向量精排）的查询耗时、召回率@10 和内存估算。数据是合成的聚类向量，分布比真实文本整齐，结论只看方向。
