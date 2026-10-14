# WMDP Unlearning Experiments

本仓库的模型训练仍仅推进 Qwen3-4B-Instruct-2507 纯 RMU；另设独立的 Qwen3Guard 外挂安全审核功能验证。维护者：HoshiuZ。

标准目标为 forget 表示误导 MSE + alpha × retain 表示保留 MSE。retain NLL 仅作监控，不加入训练目标。层号从 0 开始，hook 捕获指定 decoder layer 的输出，更新指定层的 down_proj.weight；参考模型冻结。

历史保留候选：hook12，更新层10/11/12，学习率5e-5，alpha2，steering32，40步，seed42。其历史全量成绩为 **WMDP-bio 55.3810%、MMLU 68.4874%**；MMLU相对原始基线下降2.1863个百分点，未满足此前不超过2个百分点的严格目标。历史其他配置中，严格目标下已全量评测的可行配置为hook12/alpha2/steering2/40步（Bio66.3001%、MMLU69.1355%），不混入下述本轮长度候选。

## 已有全量结果

| 实验 | hook | alpha | steering | 步数 | Bio % | MMLU % | MMLU变化/百分点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | — | — | — | 0 | 73.8413 | 70.6737 | +0.0000 |
| legacy_h7_a1_s2_80 | 7 | 1 | 2 | 80 | 59.9372 | 65.8952 | -4.7785 |
| legacy_h12_a1_s2_40 | 12 | 1 | 2 | 40 | 62.8437 | 68.3663 | -2.3074 |
| legacy_h12_a2_s2_40 | 12 | 2 | 2 | 40 | 66.3001 | 69.1355 | -1.5382 |
| h12_a2_s32_fixed40_checkpoint-40 | 12 | 2.0 | 32.0 | 40 | 55.3810 | 68.4874 | -2.1863 |
| h12_a1_s32_fixed40_checkpoint-40 | 12 | 1 | 32.0 | 40 | 50.6677 | 67.1913 | -3.4824 |

全量范围：WMDP-bio 1273题、MMLU 14042题；0-shot、无chat/system模板、BF16/SDPA，选择选项loglikelihood最大者。开发子集为Bio256题、MMLU1710题（每科30题）；来自已使用的官方测试集，不是独立验证集。调参后的全量成绩属于探索性结果。

完整参数、更新参数名称和分组变化见 [全量对比CSV](logs/rmu_scale_v2_full_comparison.csv)；所有层位置、alpha、steering、学习率及覆盖对照见 [开发集配置对比](logs/rmu_scale_v2_development_config_comparison.csv)。两种集合的成绩不可混用。保留成功、失败、短跑及中间检查点记录。

## 代码与复现入口

- 只读默认检查：`envs/unlearning/bin/python scripts/check_pure_rmu.py`，只解析纯RMU配置，不训练。
- 实现：`open-unlearning/src/trainer/unlearn/rmu.py`；其GradDiff及共享基础类属于必要依赖。上游自带算法源码保留，不代表本项目推进其他方法。
- 历史运行器：`scripts/run_rmu_adaptation.py`、`scripts/run_rmu_scale_v2.py`、`scripts/run_rmu_formal.py`；运行这些训练入口前另行明确训练范围。
- 增量重建：`scripts/rmu_adaptation_utils.py` 的 `load_delta`，按固定基模的BF16位级XOR恢复并校验参数SHA256；LFS指针须先取回对应对象。不要将指针文件直接交给模型加载器。
- 历史模型、配置及结果：`open-unlearning/saves/unlearn/`、`open-unlearning/saves/eval/`、`logs/`；基模文件指纹见 `logs/base_model_manifest.json`。初始远端不可变revision未记录，明确以本地文件和参数指纹固定版本。
- 数据、清洗/划分及缓存说明：[数据准备](docs/data-preparation.md)。受限Bio语料及其衍生缓存保留本地，不上传。
- 环境及必要检查：`logs/requirements-qwen3.txt`、`tests/`。

上轮完成清理、恢复历史、验证和整理。[清理报告](docs/pure-rmu-convergence.md)列出归档入口、LFS核实、共享依赖和实际空间变化。

基于 [OpenUnlearning](https://github.com/locuslab/open-unlearning) 修改；保留上游许可证与归属说明，详见 [上游MIT许可证](open-unlearning/LICENSE)和[上游说明](open-unlearning/README.md)。

## 40/80/120步连续训练长度对照

从固定原模型重新开始，同一条连续120步轨迹，其他配置保持历史hook12/alpha2/steering32一致。已完成训练、三个增量重建及三组全量评测。

| 本轮步数 | WMDP-bio % | MMLU % | MMLU相对基线损失/点 |
|---|---:|---:|---:|
| 40 | 55.5381 | 68.4945 | 2.1792 |
| 80 | 42.7337 | 66.9492 | 3.7245 |
| 120 | 41.1626 | 66.2085 | 4.4652 |

本轮≤2点能力损失约束下无候选；≤3点仅40步。80步是更低Bio与更大能力损失的折中，120步Bio最低；三者均互不支配，没有统一赢家。80→120步只再降低Bio1.5711点，同时再损失MMLU0.7406点，现有能力目标下不建议直接追加更长训练。

本轮40步没有与历史参数逐位复现：固定配置/数据/方向一致，同环境Flash SDPA后向非确定性探针及数值审查已留档；因此本轮40步重新全量评测，历史55.3810%/68.4874%单独列参照。未改seed或其他超参。

[中文完整报告](docs/rmu-step-comparison-results.md)、[结果JSON](logs/rmu_step_comparison_results.json)、[全量CSV](logs/rmu_step_comparison_full.csv)、[逐学科CSV](logs/rmu_step_comparison_subjects.csv)、[固定方案](docs/rmu-step-comparison-plan.md)。官方test已参与此前探索，本轮为单seed探索性结果。

本轮新增三个增量、共9个LFS对象约209.94 MiB，**仅保存在本地持久数据盘**，GitHub额度未知，尚未上传或核实远端备份。路径、SHA和真实大小见[保存状态](open-unlearning/saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120/delta_storage_status.json)；不要清理这些唯一的本地对象。公开代码、配置、日志及逐题结果正常推送；不上传受限语料、token缓存或新的完整模型。增量没有优化器状态，不能独立精确断点续训。

## Qwen3Guard 外挂审核：接入阶段已完成

独立模块位于`safety/qwen3guard/`，固定Qwen/Qwen3Guard-Gen-0.6B revision `fada3b2f655b89601929198343c94cd2f64d93cc`，直接使用官方tokenizer的输入/输出任务模板。支持原始输出、三档标签、类别、输出Refusal字段和错误记录；两种拦截策略共用一次审核，error/unknown不放行，超长输入报错而不静默截断。

已完成18条固定中英文接口样例：Safe10、Unsafe7、Controversial1，解析/生成错误0；另通过单元和超长错误路径检查。这是小规模功能验证，不是正式准确率或安全提升评测。未修改RMU训练逻辑、结果或检查点，未训练模型，未执行原模型/RMU批量生成或正式四组实验。

[阶段中文报告](safety/qwen3guard/REPORT.md)、[逐条结果](safety/qwen3guard/records/smoke_v1/results.jsonl)、[配置](safety/qwen3guard/config.json)。模型只存忽略的HF缓存，不提交Git/LFS。本阶段在接入和验证后停止；A原模型、B原模型＋Guard、C纯RMU、D纯RMU＋Guard尚未执行，下一阶段须先固定具体RMU检查点和评测协议。

WildGuardTest 独立审核评测已固定官方测试版本与数据准备规则，目前受 Hugging Face 数据集访问权限阻塞，尚未执行评测。详见 [准备与阻塞报告](safety/qwen3guard/wildguardtest/REPORT.md)。
