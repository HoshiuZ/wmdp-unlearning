# SimNPO与RMU对照产物

本目录保存受控参数范围的SimNPO、RMU加额外保留项、RMU+SimNPO联合三组实验，以及独立的CUDA实现探针。使用同一个Qwen3-4B-Instruct-2507原始起点，只更新10/11/12层down_proj，训练文章/块/顺序一致。

`checkpoint-10/20/40` 的权重是BF16 uint16 XOR+gzip增量，由Git LFS管理，真实对象保存在数据盘`.git/lfs/objects`。不是LoRA或独立完整HF模型。需先加载固定原始BF16模型，再调用`scripts/rmu_adaptation_utils.py`的`load_delta(model, checkpoint_path)`，会检查完整原始参数和重建参数指纹。未保存优化器，不能断点续训。各组配置、控制向量、采样索引、分项loss、时间、显存及逐题结果均保留。

SimNPO组不使用RMU目标或参考表示作为loss，但为了共享审计基础类仍持有冻结参考模型及未加入loss的RMU表示计算，资源数字不能当作原生SimNPO最低成本。联合实现是本项目探索，不是已验证最优方法。

方案：`docs/simnpo-rmu-plan.md`；完成报告：`docs/simnpo-rmu-results.md`。开发/全量均使用已观察官方测试数据，是探索性证据；独立文章验证仅用于loss与梯度诊断。受限原始正文与token缓存未提交。

模型许可证沿用原始Qwen3-4B-Instruct-2507的Apache-2.0，见本目录LICENSE；框架来源及MIT版权保留在上游目录和根README。SimNPO来源：https://github.com/OPTML-Group/Unlearn-Simple 。本轮先提交本地记录，未自动推送或发布模型。


后续 `joint_simhalf_screen40` 为Sim系数减半的单因素适配，10/20/40步均有无损增量，恢复方式相同。梯度诊断与结果见 `docs/simnpo-rmu-gradient-results.md`，旧三组报告保留不覆盖。诊断36层仅求梯度，没有训练扩层或修改基模。
