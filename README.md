# WMDP Unlearning Experiments

本仓库的模型训练仍仅推进 Qwen3-4B-Instruct-2507 纯 RMU；另设独立的 Qwen3Guard 分类器及四组离线防护实验。维护者：HoshiuZ。

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

[阶段中文报告](safety/qwen3guard/REPORT.md)、[逐条结果](safety/qwen3guard/records/smoke_v1/results.jsonl)、[配置](safety/qwen3guard/config.json)。模型只存忽略的HF缓存，不提交Git/LFS。接入阶段当时未执行四组实验；后续进展见下方四组离线记录，历史功能验证结果保持不变。

WildGuardTest 独立审核评测已完成：输入 1,699 条、输出 1,709 条，审核错误均为 0。主策略 Recall/FPR 分别为输入 80.37%/4.76%、输出 75.00%/4.49%；严格策略为输入 85.54%/6.46%、输出 82.04%/6.25%。本阶段只评分类器与数据集已有回答，未开展四组模型生成实验。详见 [中文评测报告](safety/qwen3guard/wildguardtest/REPORT.md)。

## 四组离线生成与审核

当前固定 A原模型、B原模型+Guard、C纯RMU80步、D同一RMU80步+Guard；80步MMLU损失3.7245点，未满足此前2点约束。[固定方案](safety/four_groups/PLAN.md)。

Safe / Controversial / Unsafe 是 Qwen3Guard 官方输出的安全等级；“主策略/main”和“严格策略/strict”是本实验自行定义的等级到动作映射，不是官方命名的两种固定策略。主策略放行 Safe/Controversial、拦 Unsafe；严格策略只放行 Safe。error/unknown 是执行或解析异常，不是官方等级，异常不放行并单列，不计为成功识别有害。两种策略共用模型、模板与审核结果，不涉及重新训练。

四组正式生成与审核记录已完成：A/C各生成1699条，失败0；B/D分别复用对应回答，输入审核复用1699条、新增输出审核2096次，审核异常0。主策略B输入/输出拦截651/4、放行1044；D为651/3、放行1045；严格策略B为706/9、放行984，D为706/5、放行988。它们仅为本实验策略的流程统计，不是独立评判的有害回答率或安全提升。

[中文结果报告](safety/four_groups/REPORT.md)、[公开汇总](safety/four_groups/results.json)、[核验记录](safety/four_groups/verification.json)。本地盲评清单3866个去重请求-回答对，对应10194条组/策略映射；无组别、模型名或Guard判断，未填人工标签。原文/回答/盲评映射只在忽略的cache目录。完成本阶段后停止，未开展独立评判。

## 512 token截断诊断

已从任一模型被512截停的请求中固定抽取40条（正常/有害各20），仅将max_new_tokens提高至1024，原模型/RMU80分别有24/23条自然结束、16/17条仍截停。未检出规则定义的高重复或控制/思考格式异常；Guard等级分别变化3/4条。新旧完整前缀仅2/40、3/40一致，抽样批内组合不同是未控制因素，不能当作严格同一回答续写或将标签变化只归因于新增尾部。建议完整回答评判前考虑更长预算，并先解决前缀一致性；本次不全量重生成，不覆盖512结果。[简短中文诊断报告](safety/truncation_diagnostic/REPORT.md)。

## WildGuard自动评判

已用官方 `allenai/wildguard` 7B 对512-token四组回答及WildGuardTest原有问答完成离线自动评分。模型固定revision `cbba4823f3e8020e5a74a5e29bf85072def6f2ff`；5588个去重请求—回答对评分5588次，其中8条pilot结果由缓存复用，3条记录因格式/生成上限记为unknown/error。WildGuardTest有害性校验：1709条中有效预测1707条，Recall67.49%、FPR2.04%、Macro-F1 85.89%；拒答校验1720条中有效1717条，Precision80.79%、Recall98.04%、Macro-F1 91.05%。

按WildGuard自动评分，有害请求上的有害回答率：A 17/754（2.25%）、B/main 3/754（0.40%）、B/strict 0/754、C 11/754（1.46%）、D/main 2/754（0.27%）、D/strict 0/754；正常请求拒答代理指标从A的94/945（9.95%）到B/main 120/945（12.70%）、B/strict 134/945（14.18%）、C 117/945（12.38%）、D/main 138/945（14.60%）、D/strict 147/945（15.56%）。纯RMU对原模型的有害回答差值置信区间含0；外挂策略显示自动评分上的下降，同时正常请求拒答代理上升。所有结论只针对512-token输出，配对区间不含评判器系统误差，不代表人工确认的有害率或完整任务帮助性。

另对V2清单做了30条ChatGPT辅助复核（具体模型版本未记录，全部 `human_verified=false`）：随机20条的18个明确有害性判断一致、2条辅助判断不确定；定向诊断10条的5个明确判断一致、5条不确定。7个B/main与D/main残留阳性案例中，2条两模型均判有害、5条辅助标签不确定；不作为人工金标准，也不改变全量WildGuard指标。[导入核验与分歧分析](safety/wildguard_judge/REPORT.md#chatgpt辅助复核)。

Safe/Controversial/Unsafe是官方输出等级；main和strict是本实验定义的动作映射，不是官方策略名。报告包含配对bootstrap区间、有效覆盖率、失败上下界、资源和限制：[中文汇总报告](safety/wildguard_judge/REPORT.md)。30条盲审材料及来源映射仅保存在本地忽略的`cache/wildguard_judge_v1/`，人工标签留空。
