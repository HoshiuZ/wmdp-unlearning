# Qwen3-4B RMU Bio — formal80 v1

Experimental derivative of the official Qwen/Qwen3-4B-Instruct-2507, trained by HoshiuZ using OpenUnlearning RMU. This is an independent research artifact, not an official Qwen release. Original model license and documentation are retained in LICENSE and BASE_MODEL_README.md; OpenUnlearning code retains its own MIT license in the repository.

Training: original weights, 80 optimizer steps, BF16/SDPA, sequence256, batch1, accumulation16, AdamW lr5e-5; gamma/alpha1, steering_coeff2, layer7 hook, update only layers5/6/7 MLP down_proj.weight. A fixed final step80 checkpoint is used. The three-step trial weights were not reused.

Held-out Bio validation and parameter/reload checks passed. Full post-training WMDP-Bio and MMLU evaluation is pending. Do not treat training loss reduction as proof of complete knowledge removal or preserved general capability. The original baseline scores are 73.8413% WMDP-Bio and70.6737% MMLU0-shot; these are not the scores of this trained checkpoint.

Model files are stored with Git LFS. Clone the repository with Git LFS installed and retrieve this model directory before loading. From the repository root:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
path = "open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_formal80_v1/model"
model = AutoModelForCausalLM.from_pretrained(
    path, dtype="bfloat16", device_map="cuda", attn_implementation="sdpa", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
```

Configuration, training/validation metrics, consumed block indices and checkpoint verification are in the parent run directory. Source corpus text/token files are not published; their access and redistribution conditions must be honored for platform delivery.
