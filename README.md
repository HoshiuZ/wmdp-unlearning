# Qwen3-4B RMU＋CE

本仓库当前只保留 **RMU＋CE、steering32、CE权重1.5倍** 的seed42/43实验。维护者：HoshiuZ。

## 配置与结果

Qwen3-4B-Instruct-2507；hook12输出（层号从0开始），更新10/11/12层的`down_proj.weight`；alpha2、steering32、CE权重约0.09004336、学习率5e-5、40步、batch1/累积16、序列上限256。

损失为：RMU遗忘表示MSE＋2×保留表示MSE＋CE权重×保留文章预测损失。CE是本项目额外加入的保留项，当前方法不含SimNPO或KL。

| 全量0-shot | WMDP-Bio | MMLU | MMLU较原始变化 |
| --- | ---: | ---: | ---: |
| 原始基模 | 73.8413% | 70.6737% | 0 |
| seed42，第40步 | 60.2514% | 68.8363% | -1.8373点 |
| seed43，第40步 | 61.1155% | 68.8150% | -1.8587点 |

两次均满足本轮MMLU下降不超过2点的目标。当前使用候选为`open-unlearning/saves/unlearn/rmu_ce_v1/seed42/checkpoint-40`，另保留两组10/20步及seed43的40步检查点。完整结果见[结果说明](docs/rmu-ce-results.md)、[JSON汇总](logs/rmu_ce_results.json)和[CSV汇总](logs/rmu_ce_results.csv)。测试集已参与前期探索。

## 运行与加载

[方法、训练、加载及评测说明](docs/rmu-ce-method.md)。可执行配置在[seed42](configs/rmu_ce/seed42.json)和[seed43](configs/rmu_ce/seed43.json)。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python scripts/run_rmu_ce.py --seed 42 --check-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python scripts/run_rmu_ce.py --seed 42
```

新运行写入`rmu_ce_v1/reproductions/`，拒绝覆盖已有目录。清理没有新增40步训练。模型采用Git LFS管理的BF16 XOR+gzip无损增量，加载需固定原始基模及`scripts/rmu_ce_utils.py`；不是独立完整HF模型，没有optimizer状态，不能断点续训。

原始基模留在`models/`，原始及处理后数据留在`data/`，两套必要基线归档留在`open-unlearning/saves/eval/`。基模文件SHA见[清单](logs/base_model_manifest.json)，训练数据版本与SHA见[块清单](logs/rmu_formal_blocks_manifest.json)，数据处理规则见[数据说明](docs/data-preparation.md)。

## 当前目录与历史

其他项目实验的模型、结果、专用脚本和文档已从当前版本清理；Git历史保留，历史LFS对象仍在远端。本地只保留当前产物所需LFS缓存。本地释放约20.41 GiB、当前可用约37.30 GiB。清理范围见[清单](logs/method_cleanup_manifest.json)和[清理说明](docs/method-cleanup.md)，新入口的损失/梯度及六个检查点重建核查见[验证记录](logs/rmu_ce_verification.json)。

保留产物中的原始运行配置和日志是当时执行记录，其中旧handler名称及路径不作篡改。当前运行使用独立`RMUCE`入口，旧路径迁移关系见[映射](reference/rmu_ce/path_migration.json)，不再依赖已删除实验。`open-unlearning/`保留通用上游框架及其依赖；项目实验入口仅RMU＋CE。

阶段检查通过后及时提交并推送GitHub。根目录私人工作文档、原始基模、环境及受限Bio正文/token缓存不上传。历史版本可从Git及Git LFS取回。

## 来源与许可

框架基于CMU Locus Lab的[OpenUnlearning](https://github.com/locuslab/open-unlearning)，起始版本`17cbbc87192e6934deb92875c359c91bbd837fb4`，见[版本记录](logs/code_commit.txt)。保留[MIT许可证](open-unlearning/LICENSE)（Copyright (c) 2025 CMU Locus Lab）和[上游README](open-unlearning/README.md)。通用框架包含其他算法源码，未作为本项目当前实验入口。

RMU/WMDP来源：[官方项目](https://github.com/centerforaisafety/wmdp)、[论文](https://arxiv.org/abs/2403.03218)。评测使用[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。基模为[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，模型许可证见[Apache-2.0](open-unlearning/saves/unlearn/rmu_ce_v1/LICENSE)。本项目贡献为模型接入、RMU＋CE组合实现及实验记录；不是原论文标准RMU复现。数据许可按各自来源执行，受限语料不公开再分发。
