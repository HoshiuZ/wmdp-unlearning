# RMU80 终端聊天

在项目根目录运行：

```bash
./envs/unlearning/bin/python scripts/chat_rmu80.py
```

输入问题后等待回答即可继续聊天。输入 `/clear` 清空多轮上下文，输入 `/exit` 或按 `Ctrl+C` 退出。默认每轮最多生成 1024 个新 token；可用 `--max-new-tokens 256` 等参数调整。

脚本只使用本地 `Qwen3-4B-Instruct-2507` 基模与 RMU80 无损增量，启动时校验两者指纹并在内存中恢复模型。需要可用的 CUDA/BF16 GPU；不加载安全分类器、不联网下载模型，也不保存聊天记录。加载或校验失败时会报错退出，不会改用原模型。
