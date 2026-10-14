# 方法清理记录

按用户明确授权，只保留RMU＋CE steering32、CE权重1.5倍的seed42/43实验，删除其他实验产物及项目专用代码、文档。

保留六个10/20/40步无损增量、两套全量结果、训练日志与完整原始配置；产物统一到rmu_ce_v1/。保留原始基模、数据、两套必要基线和通用上游框架。根目录私人工作文档不参与Git上传或本轮删除。

删除旧实验文件约7.76 GiB，删除181个不用的本地LFS缓存对象约12.65 GiB，合计约20.41 GiB；数据盘可用约37.30 GiB。保留18个活跃LFS对象，约376 MiB。精确清单及路径见logs/method_cleanup_manifest.json。

历史提交未重写。删除前git lfs push --all origin main成功确认历史可达LFS备份；远端历史LFS不删除，旧版本可按Git历史取回。当前分支中其他实验已删除。

新训练入口scripts/run_rmu_ce.py、新评测入口scripts/evaluate_rmu_ce.py，当前配置configs/rmu_ce/。迁移先解除旧实验依赖，再删除；原始运行日志/配置作为当时事实保留，旧路径有reference/rmu_ce/path_migration.json映射。

验证包含：生产BF16重复旧实现自身存在梯度数值波动；在确定性math SDPA与确定性CUDA下，新旧RMU＋CE两步损失、三个参数梯度逐位一致。验证后恢复原始参数指纹，六个保留检查点均逐位重建成功，逐题记录的hash/答案/计数与原始基线核对一致。删旧文件后的两个入口--help均成功，当前本地模块依赖解析通过，数据SHA复核通过，裁剪后的Git LFS fsck通过。

没有新的40步训练或全量评测，没有改变已保存权重。最终清理记录与代码按授权推送GitHub。
