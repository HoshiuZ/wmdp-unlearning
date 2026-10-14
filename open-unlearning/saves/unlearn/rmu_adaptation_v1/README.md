# RMU 适配探索产物

每个目录是独立原始模型起点。run.json包含实际配置、版本、耗时、显存和核查，training_config.json是解析配置，training_metrics.jsonl是逐步训练及语料验证，consumed_block_indices.json是受限缓存中的实际采样索引。checkpoint-10/20/40保存三个更新参数的无损增量及开发评分。原始语料和token缓存仍按来源权限留本地。

*.xor.gz是原始BF16参数与训练参数的uint16逐位XOR再gzip压缩，不是浮点减法、LoRA或独立完整HF模型。工作树可能是Git LFS pointer，真实文件保存在LFS cache/远端；加载helper支持本地默认LFS cache或已物化的文件。需要原始基模和原基模完整参数指纹记录，所有重建均验证逐位一致。优化器状态不保存，不作为断点续训包。

完整比较见根仓库docs/rmu-adaptation-results.md和logs/rmu_adaptation_comparison.csv。开发题来自曾经全量评测的官方test，最终全量分数为探索性，不是独立未参与调参的测试。阈值2个百分点只是本轮目标。

从根仓库加载最终选定增量（有本地原模型、Git LFS对象或已物化增量时）：

```python
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from rmu_adaptation_utils import load_delta
selection = json.loads((root / "open-unlearning/saves/unlearn/rmu_adaptation_v1/selected_checkpoint.json").read_text())
base = root / "models/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
    attn_implementation="sdpa", device_map="cuda", local_files_only=True)
load_delta(model, root / selection["checkpoint"])
model.eval()
tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
```

基模Qwen/Qwen3-4B-Instruct-2507为Apache-2.0；模型许可证保留为本目录LICENSE。原始远端revision未在最初下载记录，不能虚构；本地文件和完整参数指纹锁定。上游框架归CMU Locus Lab，MIT声明见open-unlearning/LICENSE，算法/基准来源见根README。
