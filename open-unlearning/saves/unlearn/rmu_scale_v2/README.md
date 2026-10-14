# 第二轮表示尺度与数据覆盖产物（本地留档）

本目录保留3组超参数实验、1组覆盖对照、所有开发结果和最多2次全量探索性评测。本轮未发布模型或推送远端。具体结论见 [中文报告](../../../../../docs/rmu-scale-v2-results.md)。

这些 `.xor.gz` 是 Git LFS 管理的 BF16 uint16 XOR+gzip 增量，不是 LoRA，也不是独立 Hugging Face 模型。真实对象在数据盘 `.git/lfs/objects`；工作树可能仅有 pointer。每个 `checkpoint.json` 记录原始及训练参数指纹、压缩对象 SHA256 和大小。仅改变层10/11/12的 `mlp.down_proj.weight`，其他参数保持原始值。

第一组 `h12_a1_s16_fixed40` 的10/20步目录只有指标；仅40步有可重建权重。另3组保存10/20/40步。未保存优化器状态，不能声称断点续训。

从仓库根目录使用环境中的 Python 加载（不联网）：

```python
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, "scripts")
from rmu_adaptation_utils import BASE, ROOT, load_delta

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, local_files_only=True,
    attn_implementation="sdpa", device_map="cuda",
)
tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
load_delta(model, ROOT / "open-unlearning/saves/unlearn/rmu_scale_v2/h12_a2_s32_fixed40/checkpoint-40")
model.eval()
```

加载必须从 `logs/base_model_manifest.json` 与历史 `base_weight_fingerprints.json` 对应的 Qwen3-4B-Instruct-2507 原始模型开始，helper核验全部原始参数与重建后的3个张量。跨机器需保留基模和实际LFS对象，不能只复制pointer。

训练配置、控制向量、实际文章索引、覆盖组位置记录、运行时间及峰值显存见各组 `run.json` 和日志。数据来源与处理版本见 `logs/rmu_formal_blocks_manifest.json`、`logs/rmu_scale_v2_coverage_manifest.json`；受限Bio文章/token缓存没有公开提交。

模型来源与许可继承 Qwen3-4B-Instruct-2507 的 Apache-2.0（原始许可证在 `models/Qwen3-4B-Instruct-2507/LICENSE`，下载的基模留本地）；代码来源与上游MIT版权见仓库根README及 `open-unlearning/LICENSE`。结果是已经观察过官方test后的探索性证据，不证明知识彻底删除。
