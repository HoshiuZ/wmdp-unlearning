# 唯一保留方法：RMU＋CE

Qwen3-4B-Instruct-2507；hook为第12层输出（0-based），只更新10/11/12层的down_proj.weight，共74,711,040个参数。损失为RMU遗忘表示MSE＋2倍保留表示MSE＋0.09004336292634454倍保留文章CE。CE是本项目额外增加的保留项；不包含SimNPO或KL。参数与此前CE权重1.5倍的有效训练目标相同。

学习率5e-5、AdamW、constant scheduler、裁剪1、batch1/累积16、40步，序列上限256。CE按每样本有效预测token取均值；排除padding和labels=-100，内部表示MSE也排除掩码位置。参考模型冻结、eval且no_grad。控制向量L2范数32，每次运行固定；seed42/43归档方向、640次每角色采样和实际token预算存于reference/rmu_ce/。

## 训练

使用现有环境，不安装依赖：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python scripts/run_rmu_ce.py --seed 42 --check-only
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python scripts/run_rmu_ce.py --seed 42
```

新运行写入rmu_ce_v1/reproductions/，已有目录拒绝覆盖；清理不触发新的40步训练。seed43对应配置也保留。完整可执行配置在configs/rmu_ce/seed42.json和seed43.json。

## 加载与评测

检查点为BF16 uint16 XOR+gzip无损增量，需固定原始基模。不是独立完整HF模型，无optimizer状态，不能断点续训。

```python
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
import torch
from transformers import AutoModelForCausalLM
from rmu_ce_utils import BASE, OUT, load_delta
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16,
    local_files_only=True, attn_implementation='sdpa', device_map='cuda')
load_delta(model, OUT / 'seed42/checkpoint-40')
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 envs/unlearning/bin/python scripts/evaluate_rmu_ce.py --seed 42 --output open-unlearning/saves/unlearn/rmu_ce_v1/evaluation_recheck_seed42
```

评测使用原始归档0shot请求，Bio1273/MMLU14042，batch1、BF16/SDPA、无聊天模板，四选项loglikelihood。开发集256/1710用于留档，本配置第40步在评测前固定。原始基模评测归档属于必要基线，保留在saves/eval/，不参与本轮删除。

## 路径和历史

保留seed42/seed43各10/20/40步及两组全量结果。旧目录迁移关系见reference/rmu_ce/path_migration.json。产物中的原始training_config.json、run.json及原始日志保留当时的handler名称、代码提交和路径，属于历史执行记录；新执行入口使用独立RMUCE类和configs/rmu_ce/，不再依赖旧实验目录。Git历史保留，历史LFS对象仍在GitHub，可按历史版本取回；本地只缓存保留产物需要的LFS对象。
