# ep25 微调 / RAG / 提示词 成本与决策计算器

按三种典型场景估算一年成本，并用启发式规则给出倾向（高 churn→RAG，强风格+稳知识→微调，小流量→纯提示）。

价格快照 2026-10-30：OpenAI gpt-4.1 微调训练与推理价、gpt-6.1-sol 基线、text-embedding-3-large。不调用 API。

相关论文（文章正文引用）：Ovadia et al. 2023 arXiv:2312.05934；Balaguer et al. 2024 arXiv:2401.08406；Hu et al. 2021 LoRA arXiv:2106.09685。

## 运行

```bash
python3 decision_cost.py
```

结果写入 `outputs/ep25_results.json`。可改 `DEFAULTS` 里的场景参数。
