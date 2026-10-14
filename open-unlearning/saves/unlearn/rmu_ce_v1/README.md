# RMU＋CE保留产物

只保留steering32、alpha2、CE权重0.09004336292634454、40步的seed42与seed43。每组有checkpoint-10/20/40和完整训练记录；full_evaluations/保存各自全量评分；checks/保存实现检查。

seed42第40步：Bio60.2514%、MMLU68.8363%（较基线-1.8373点）。seed43第40步：Bio61.1155%、MMLU68.8150%（-1.8587点）。

权重为Git LFS管理的无损BF16 XOR+gzip增量。使用scripts/rmu_ce_utils.py的load_delta，固定原始基模与指纹，不是LoRA/完整HF模型。未保存optimizer，不能断点续训。

训练/加载/评测见docs/rmu-ce-method.md，当前结果见docs/rmu-ce-results.md。原始运行日志和配置中的旧路径由reference/rmu_ce/path_migration.json映射到当前目录。许可证见本目录LICENSE（原始Qwen Apache-2.0）；上游框架版权见open-unlearning/LICENSE。
