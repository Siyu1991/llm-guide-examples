# ep06 RAG 从零到能用

对应专栏第 6 期。读 Markdown → 按标题切块 → bge-small-zh-v1.5 向量化 → 检索 → 拼提示 → 生成。

```bash
pip install -r requirements.txt
python rag.py eval                                  # 24 道题评估检索命中率
python rag.py ask "去上海出差，酒店一晚最多能报多少钱？"
python rag.py ask "你的问题" --kb /path/to/your/markdown_dir
```

生成部分：
- 设置 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`、`RAG_MODEL`）时调用 OpenAI 兼容接口；
- 否则使用离线抽取式兜底：从检索片段中挑一句最相似的原文返回。它不会推理，经常挑错句子，只用于跑通流程。

`kb/` 是虚构公司"青禾科技"的示例文档，故意放了一份已废止的旧版报销制度，第 7 期会用到。峰值内存约 0.4GB。
