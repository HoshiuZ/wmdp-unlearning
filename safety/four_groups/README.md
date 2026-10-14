# 四组离线生成与审核

先读 [固定方案](PLAN.md)，参数与来源见 config.json / manifest.json，试运行见 pilot_summary.json，正式汇总见 results.json / REPORT.md。仅原模型与RMU80两个生成模型；不训练、不做独立评判。

Safe/Controversial/Unsafe是官方等级；main/strict是本实验定义的动作映射，不是官方命名的策略。error/unknown单列异常、不放行。具体定义与统计边界见方案。

复现顺序（会实际生成，需授权范围及本地固定资产齐备）：

```bash
envs/unlearning/bin/python -m safety.four_groups.prepare
envs/unlearning/bin/python -m unittest safety.four_groups.test_pipeline
envs/unlearning/bin/python -m safety.four_groups.generate --model original --phase pilot
envs/unlearning/bin/python -m safety.four_groups.generate --model rmu80 --phase pilot
envs/unlearning/bin/python -m safety.four_groups.audit --phase pilot
envs/unlearning/bin/python -m safety.four_groups.finish --phase pilot
envs/unlearning/bin/python -m safety.four_groups.run_full
envs/unlearning/bin/python -m safety.four_groups.report
```

实际使用现有envs/unlearning、OMP_NUM_THREADS=2、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1。先后启动工作进程，禁止同时驻留两个生成模型。run_full只按原模型→RMU→Guard→记录顺序推进，任何工作进程失败即停止。完成后不启动独立评判。

SQLite事务逐条保存结果；相同协议续跑会跳过已记录结果（包括错误，不擅自重试），中断批次最多重做尚未落盘样本。config和关键模型/模板/审核指纹进入协议哈希及缓存键，变更不能静默复用。状态JSON原子替换，SQLite是回答事实来源。

全部原文、回答、原始审核输出、系统逐条记录及盲评映射只在已Git忽略的cache/four_groups_v1。盲评文本文件不能与mapping一起给评判者，不包含Guard结果，不填预设安全标签。相同请求和完全相同回答复用同一blind_id，并保留全部来源映射。

正式结束后只提交本目录公开代码/配置/汇总/哈希，以及README和历史报告的策略说明；禁止提交cache、权重或原始文本。
