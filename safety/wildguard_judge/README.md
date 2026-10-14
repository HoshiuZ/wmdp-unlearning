# WildGuard 自动评判

本目录使用官方 `allenai/wildguard` 7B 对已经生成的 512-token 四组回答做离线自动评分，也用 WildGuardTest 原有人工标签校验评判器。它不参与 Qwen3Guard 防护、不生成新回答、不训练模型。本轮已完成，配置、汇总指标、对照区间和限制见 [中文报告](REPORT.md)。

评判器 revision 固定为 `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`，Apache-2.0；访问审批已生效，权重于 2026-10-15 下载校验。共保存必要文件约 14.50 GB，只下载 BF16 safetensors，未下载重复 `.bin` 权重。当前环境为 Python 3.11.17、PyTorch 2.9.1+cu128、Transformers 5.5.4；为加载 Qwen tokenizer，在既有环境仅添加 `sentencepiece==0.2.1`，未升级其他包。

采用官方模型卡输入模板、官方三字段解析和确定性 BF16/CUDA/SDPA 推理。先对8条真实数据完成模板与字段验证，再对5588个去重请求—回答对逐条评分；8条 pilot 已由同一缓存复用。模型预测标记为 `wildguard_auto`，人工标签独立留空。

重跑命令（使用已准备的本地数据和已下载模型）：

```bash
envs/unlearning/bin/python -m safety.wildguard_judge.verify
envs/unlearning/bin/python -m safety.wildguard_judge.run --phase pilot
envs/unlearning/bin/python -m safety.wildguard_judge.run --phase full
envs/unlearning/bin/python -m safety.wildguard_judge.summarize
```

每条预测在本地事务缓存中单独落盘，`--phase full` 可断点续跑。原始请求、回答、逐条分数、来源映射与人工盲审材料均保存在 Git 忽略的 `cache/wildguard_judge_v1/`；Git 只保存模型来源元数据、配置、代码和不含原文的汇总报告。
