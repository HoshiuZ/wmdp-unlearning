# 512 token截断诊断

只对固定40条请求生成原模型/RMU80的1024版本，不全量重生成、不训练、不开展人工标注。若REPORT.md与results.json已存在，run会直接退出，避免重复运行。

```bash
OMP_NUM_THREADS=2 envs/unlearning/bin/python -m safety.truncation_diagnostic.prepare
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python -m safety.truncation_diagnostic.run
```

配置与固定样本ID见manifest.json。推理配置从四组CFG复制，只改max_new_tokens；源配置中的local_root是历史字段，诊断输出不使用该字段。实际输出统一放在Git忽略的cache/truncation_diagnostic_v1，新SQLite/回答/审核结果/逐条比较均与源四组分离。源四组数据库通过SQLite mode=ro打开，不写回。顺序加载原模型、RMU80、Guard，既有增量仅在内存恢复，不保存完整权重。

每条1024生成和审核结果都事务落盘，缓存键包含诊断协议/内容/任务/模型；失败不默认安全，连续5次审核失败停止。报告只公开汇总与清单哈希，不公开原文、回答、原始Guard输出或逐条敏感记录。

batch20和其余推理配置均不变，但抽样后的批内组合与历史生成不同；BF16 greedy可能因此改变前缀，报告明确区分这一干扰，不能将前缀改变完全解释为上限影响。旧记录没有原始token IDs，比较原文前缀及新结果前原token数的解码文本，不声称原始token IDs逐位相同。

文本检查是预先定义的自动提示规则，不是人工标注或独立安全评判。Guard是同一个既有分类器，只用于比较两版本审核标签/动作；此处对所有40条的两个模型/两个版本补足输出审核，不依赖主流程是否输入拦截。main/strict是本实验映射，不是官方策略名称。
