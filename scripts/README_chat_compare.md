# 原模型与 RMU80 终端对比

在项目根目录运行：

```bash
./envs/unlearning/bin/python scripts/chat_compare.py --max-new-tokens 128
```

每次输入一个问题，程序用同一官方聊天模板依次生成原模型和 RMU80 的回答；每题互相独立，不带入历史。普通单行问题直接输入。多行问题输入 `/multi`，粘贴内容后单独输入 `/send` 提交；`/cancel` 取消，`/exit` 或 `Ctrl+C` 退出。

程序只读取本地基模和连续120步RMU轨迹的checkpoint-80，不加载安全分类器、不下载模型、不保存聊天记录。启动时完整校验两种模型状态；同一GPU模型实例只缓存和切换增量涉及的参数，每次切换都会核验这些参数的指纹。
