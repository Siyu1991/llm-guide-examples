# ep27 提示注入与权限过滤

玩具 Agent：从文本中解析 `<tool>name{json}</tool>`，按角色 ACL 决定是否放行。用例覆盖直接注入、间接注入（文档夹带）、合法读、只读角色拒写。

```bash
python3 agent.py
python3 -m unittest test_agent -v
```

说明：agent 角色对 `refund_order` 的 ACL 放行，是为了演示「注入成功解析后仍可能通过权限表」——生产写操作还需确认闸，不能只靠提示词。
