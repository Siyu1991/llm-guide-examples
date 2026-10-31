# ep18 GitHub Copilot 额度与会话成本

按官方「GitHub AI Credits」规则，估算短问答 / 长 Agent 会话会吃掉多少额度。

数字来源（2026-10-30 查阅）：

* https://docs.github.com/en/copilot/concepts/billing-and-usage/individuals/billing
* https://docs.github.com/en/copilot/reference/copilot-billing/models-and-pricing
* https://docs.github.com/en/copilot/get-started/plans

不调用 API，不需要密钥。会话 token 数为作者构造的用量假设。

## 运行

```bash
python3 copilot_econ.py
python3 test_copilot_econ.py
```
