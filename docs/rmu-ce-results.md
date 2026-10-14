# 保留方法结果

只保留RMU＋CE、steering32、CE权重1.5倍的两组产物。

| seed | WMDP-Bio | MMLU | MMLU相对原始变化 |
| --- | ---: | ---: | ---: |
| 42 | 60.2514% | 68.8363% | -1.8373点 |
| 43 | 61.1155% | 68.8150% | -1.8587点 |

原始基线Bio73.8413%、MMLU70.6737%。两个seed都满足本轮MMLU下降不超过2点的目标；MMLU平均下降1.8480点。具体使用候选为rmu_ce_v1/seed42/checkpoint-40；两组完整逐题评分、训练配置、分项损失、实际采样和控制向量均保留。

本配置是前期探索后固定的方案，官方测试已参与前期探索，不作为独立未观察测试。用户已决定清理其他实验；当前目录不再保存它们的模型、对比表或专用实现，历史版本保留在Git中。

当前汇总见logs/rmu_ce_results.json/.csv，清理和重构核查见logs/method_cleanup_manifest.json及logs/rmu_ce_verification.json。
