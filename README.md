# WMDP Unlearning Experiments

本仓库记录基于 Qwen3-4B-Instruct-2507 的文本有害知识遗忘实验，保存实验代码、配置、日志、评测结果及后续交付产物。维护者：HoshiuZ。

## 上游来源与致谢

训练与评测框架基于 CMU Locus Lab 的 [OpenUnlearning](https://github.com/locuslab/open-unlearning)，代码位于 `open-unlearning/`。本项目采用的上游起始提交为 [`17cbbc87192e6934deb92875c359c91bbd837fb4`](https://github.com/locuslab/open-unlearning/tree/17cbbc87192e6934deb92875c359c91bbd837fb4)，也记录在 [logs/code_commit.txt](logs/code_commit.txt)。该编号是上游代码版本，不是模型或数据版本。

OpenUnlearning 使用 MIT License，原始版权声明为 `Copyright (c) 2025 CMU Locus Lab`。本仓库保留了完整的 [上游许可证](open-unlearning/LICENSE) 和 [上游 README](open-unlearning/README.md)。上游框架与已有算法的贡献归原作者，本项目的工作是模型接入、实验配置调整、评测修复以及实验与交付记录。模型权重和数据的许可分别以各自来源为准。

相关项目与研究来源：

- OpenUnlearning：[项目](https://github.com/locuslab/open-unlearning)、[技术报告](https://arxiv.org/abs/2506.12618)。
- WMDP 与 RMU：[官方项目](https://github.com/centerforaisafety/wmdp)、[论文](https://arxiv.org/abs/2403.03218)。
- 选择题评测工具：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- 起始模型：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)。

使用本仓库产出研究报告时，请同时引用实际使用的框架、方法、基准及模型；OpenUnlearning 的引用信息见上游 README 的 “Citing this work” 部分。

## 本项目已完成的工作

截至 2026-10-13（Asia/Shanghai）：

- 添加本地 Qwen3-4B-Instruct-2507 模型配置，使用 BF16 和 SDPA。
- 修复评测器读取 `simple_evaluate_args` 的配置传递问题，并用20题运行确认 `limit` 生效。
- 完成训练前全量 WMDP-Bio 评测，保存逐题记录、汇总、日志与 Hydra 配置。
- 保存依赖快照、上游提交号、代码修改补丁及模型配置。

训练前 WMDP-Bio 基线：1273题，Accuracy **73.8413%**，标准误约 **1.2323个百分点**。评测为0-shot、选择题 loglikelihood 评分、无聊天模板、无额外 system instruction。

原始证据位于 [正式基线目录](open-unlearning/saves/eval/qwen3_4b_base_bio_full_v1/)。历史 smoke 目录也保留：`qwen3_4b_base_bio_smoke20` 因修复前参数未生效实际为全量运行；`qwen3_4b_base_bio_smoke20_fixed` 才是20题验证。

## 尚未完成的工作

已下载官方 S3 语料及获批的 Bio 遗忘语料，完成文件格式、样本数和校验值核查。Bio 遗忘语料共24,453条，为带 `text` 等字段的 Parquet；S3语料为每行一个JSON字符串。MMLU 训练前正常能力基线已完成：57个学科、14,042题，0-shot Accuracy **70.6737%**（9,924题正确），标准误约 **0.3617个百分点**。正式产物见 [MMLU基线目录](open-unlearning/saves/eval/qwen3_4b_base_mmlu_0shot_full_v1/)，协议和核查记录见 [logs/mmlu_baseline_run.json](logs/mmlu_baseline_run.json)。Bio 数据清洗与文章组划分已完成（seed42，约95/5）：遗忘训练/验证23,220/1,217条，保留训练/验证57,488/3,029条，规则和局限见 [数据处理记录](docs/data-preparation.md)。RMU 短跑与正式训练、训练后评测和完整交付包均尚未完成。当前结果仅是原始模型基线，不表示已完成知识遗忘。后续 WMDP 分数下降也只能说明该代理任务能力减弱，需同时检查正常能力保持。

## 仓库内容与留档

- `open-unlearning/`：上游源码、本项目改动、实验配置及评测产物。
- `logs/`：环境快照、代码来源和配置留档。
- `data/`：原始数据与MMLU评测快照；获批的 Bio 遗忘语料不公开提交。`outputs/`：后续产物目录，目前为空。
- 后续 `{model_id}_delivery/`：模型信息、实际数据、训练配置/指标、前后评测、模型产物及附件。

根目录的私人工作文档不上传；公开入口 `README.md` 单独放行。环境、缓存和下载的原始基模不纳入 Git。交付目录内的文档、数据和日志应保存，不能只提供截图或总分。完整权重配置了 Git LFS 规则，但服务器尚未安装 Git LFS；上传权重前需要安装并确认文件以 LFS 对象提交。

当前配置含 AutoDL 服务器本地路径，复现时需调整模型和数据位置。环境依赖见 [logs/environment.txt](logs/environment.txt)，依赖调整文件见 [logs/requirements-qwen3.txt](logs/requirements-qwen3.txt)，已有评测修复见 [logs/code_changes.patch](logs/code_changes.patch)。RMU 训练路径尚未验证。

Bio 遗忘语料访问条款限制未经许可向协作者名单之外分发；原始文件保存在本地，GitHub仅记录来源、固定版本、文件清单及统计，不包含token或受限正文。平台交付的接收方权限需另行确认。

MMLU评测数据固定为 `cais/mmlu@c30699e8356da336a370243923dbaf21066bb9fe`，从官方 all/test 与 all/dev 按subject分组读取，不改写题目。完整结果保留任务配置、随机种子、版本、样本统计及逐题记录。0-shot结果不能直接与公开5-shot排行榜比较。

RMU 实现和短跑输入已核查，准备了8+8训练文章、256token、3步配置；修复了梯度冻结和batch损失广播问题，小模型CPU检查通过。见 [RMU准备记录](docs/rmu-readiness.md)。真实4B CUDA三步短跑现已通过：loss/梯度有限，仅指定三组权重改变；保存重载指纹一致且forward有限，峰值已分配显存约18.11GiB。正式训练及后训练评测尚未执行。
