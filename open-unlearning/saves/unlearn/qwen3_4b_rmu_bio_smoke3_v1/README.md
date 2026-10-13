# 三步短跑模型：无损增量保存

为释放数据盘空间，2026-10-14（北京时间）将三步短跑完整BF16权重转为 `delta_checkpoint/` 的XOR+gzip增量。转换前核验完整训练参数指纹、所有原权重分片文件SHA256；保存后重建的3个更新参数逐位一致，其余参数与原始模型指纹一致。

9个冗余完整safetensors分片已移除。`model/` 仅保留历史配置、tokenizer和索引，不能再直接作为完整模型from_pretrained加载。旧 `model_artifacts.json` 是转换前的历史清单，不表示完整分片仍在本地；新状态与移除文件清单见 `logs/smoke_compaction_20261014.json`。指纹、配置、控制向量、日志及短跑验证记录均保留。

恢复方式：从固定原始 `models/Qwen3-4B-Instruct-2507` 加载BF16模型，然后调用 `scripts/rmu_adaptation_utils.py` 中的 `load_delta(model, ROOT / "open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/delta_checkpoint")`。需要真实Git LFS对象，不是仅pointer；未保存优化器状态，不能用于断点续训。
