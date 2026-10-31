# llm-guide-examples

「代码与模型之间」专栏配套示例。各目录可独立运行。

## 目录

| 目录 | 对应篇 | 说明 |
|------|--------|------|
| `ep18_copilot_econ` | 18｜Copilot 产品拆解 | 按官方 AI Credits 表折算会话用量 |
| `ep19_memory_v2` | 19｜应用层记忆 | 写入 / 检索 / 冲突 / 遗忘的最小实现 |
| `ep20_claim_check` | 20｜附篇｜主张核查 | CSV 主张 + 证据 → 通过 / 失败 / 未知 |
| `ep21_repo_score` | 21｜附篇｜开源采纳打分 | 对本地仓库元数据打试点分 |
| `ep22_ai_handoff` | 22｜附篇｜AI 编程交付 | 验收条件与交接 CSV |

更早的 `ep01`–`ep17`、`ep23`–`ep25` 等目录若存在，保持原样。

## 运行

需要 Python 3.10+。各目录下：

```bash
cd <目录>
python -m pytest -q   # 若有 tests/
python main.py        # 或 README 里写的入口
```

不依赖云端 API Key；输入均为本地 fixtures。

## 许可

与专栏示例一致，仅供学习。
