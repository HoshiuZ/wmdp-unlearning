# WMDP Unlearning Experiments

本仓库当前只推进 Qwen3-4B-Instruct-2507 的纯 RMU 实验。维护者：HoshiuZ。

标准目标为 forget 表示误导 MSE + alpha × retain 表示保留 MSE。retain NLL 仅作监控，不加入训练目标。层号从 0 开始，hook 捕获指定 decoder layer 的输出，更新指定层的 down_proj.weight；参考模型冻结。

当前保留候选：hook12，更新层10/11/12，学习率5e-5，alpha2，steering32，40步，seed42。其全量成绩为 **WMDP-bio 55.3810%、MMLU 68.4874%**；MMLU相对原始基线下降2.1863个百分点，未满足此前不超过2个百分点的严格目标。严格目标下已全量评测的可行配置为hook12/alpha2/steering2/40步（Bio66.3001%、MMLU69.1355%）。

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

本轮仅清理、恢复历史、验证和整理，没有启动新训练或重复全量评测。[清理报告](docs/pure-rmu-convergence.md)列出归档入口、LFS核实、共享依赖和实际空间变化。

基于 [OpenUnlearning](https://github.com/locuslab/open-unlearning) 修改；保留上游许可证与归属说明，详见 [上游MIT许可证](open-unlearning/LICENSE)和[上游说明](open-unlearning/README.md)。
