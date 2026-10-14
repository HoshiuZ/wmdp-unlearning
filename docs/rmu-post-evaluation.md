# 正式 RMU 模型训练后评测

评测对象是预先固定的第80步模型 `qwen3_4b_rmu_bio_formal80_v1`，直接重新加载已保存的完整权重。未按测试结果选择检查点或调整训练参数。

## 复现

从仓库根目录运行（已有独立环境及本地模型）：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 envs/unlearning/bin/python scripts/run_rmu_post_eval.py --base-check
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 envs/unlearning/bin/python scripts/run_rmu_post_eval.py
envs/unlearning/bin/python scripts/summarize_rmu_post_eval.py
```

脚本内部切换到 `open-unlearning/`，复用项目评测器；完整配置、任务配置、样本数量、逐题响应、汇总和控制台日志均保留。已存在的结果默认跳过，避免覆盖基线及重复正式测试。跨机器需调整本地模型路径，任务数据路径使用仓库相对路径。BF16、SDPA、batch1、0-shot、无聊天模板、无额外system instruction，A/B/C/D选择题loglikelihood评分；种子显式为0/1234/1234/1234。

## 数据与可比性

WMDP-Bio从原始基线1273条逐题记录重建本地Parquet快照，保持内容和doc_id顺序，避免下载版本漂移。来源是原基线官方 `cais/wmdp` 测试题；原始记录中的revision为 `7125571f22f032c56415e7980f48d877dd830ff8`，本次并未额外通过远端验证该revision。快照SHA256保存在评测配置中。任务采用当前安装lm-eval的官方Bio描述和评分模板，只将数据读取改为本地Parquet。

MMLU使用先前固定的官方 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe` 本地快照及57个学科配置，共14042题。所有训练后题目和四个评分请求逐条与基线做相等核查，不能只凭题量相同判断可比。

原模型先抽查Bio前16题：预测全部与原基线一致；这只是评测封装兼容性验证，不是新的正式基线，也不参与参数选择。全量比较报告见 `logs/rmu_post_eval_summary.json`。

## 结果解释范围

WMDP准确率下降表示该选择题代理任务上的目标能力减弱，不证明知识彻底删除。MMLU只反映该基准上的正常能力；同时报告其损失，不将正常能力下降掩盖为选择性遗忘。独立保留验证块的NLL/PPL已在训练中记录，但不代表广泛通用文本能力。第一版尚未开展恢复攻击、再学习实验或多随机种子统计，不编造安全总分或现实风险概率。

后续按Word要求整理标准交付目录，明确数据许可、受限数据的接收方授权及实际缺项。

## 全量结果

- WMDP-Bio：940/1273（73.8413%）→763/1273（59.9372%），变化-13.9042个百分点。
- MMLU：9924/14042（70.6737%）→9253/14042（65.8952%），变化-4.7785个百分点。
- 两项均0-shot、BF16/SDPA、batch1、无聊天模板/system；全部题目及评分请求与基线逐条一致。原模型Bio16题抽查评分最大差0。固定80步模型，未按测试调参或选择checkpoint。
- scripts/run_rmu_post_eval.py与scripts/summarize_rmu_post_eval.py、固定Bio测试快照、任务配置、完整/逐题JSON、日志及logs/rmu_post_eval_summary.json均留档。目标代理能力减弱同时存在正常能力代价，不宣称知识彻底删除。
- 下一步按Word整理九个必需文件及完整标准交付目录；受限数据授权范围与保留语料再分发许可仍需明确，不公开受限正文或token。
