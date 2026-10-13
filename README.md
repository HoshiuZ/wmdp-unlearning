# WMDP Unlearning Experiments

本仓库记录基于 Qwen3-4B-Instruct-2507 的文本有害知识遗忘实验，保存实验代码、配置、日志、评测结果及后续交付产物。维护者：HoshiuZ。

## 上游来源与致谢

训练与评测框架基于 CMU Locus Lab 的 [OpenUnlearning](https://github.com/locuslab/open-unlearning)，代码位于 `open-unlearning/`。本项目采用的上游起始提交为 [`17cbbc87192e6934deb92875c359c91bbd837fb4`](https://github.com/locuslab/open-unlearning/tree/17cbbc87192e6934deb92875c359c91bbd837fb4)，也记录在 [logs/code_commit.txt](logs/code_commit.txt)。该编号是上游代码版本，不是模型或数据版本。

OpenUnlearning 使用 MIT License，原始版权声明为 `Copyright (c) 2025 CMU Locus Lab`。本仓库保留了完整的 [上游许可证](open-unlearning/LICENSE) 和 [上游 README](open-unlearning/README.md)。上游框架与已有算法的贡献归原作者，本项目的工作是模型接入、实验配置调整、评测修复以及实验与交付记录。模型权重和数据的许可分别以各自来源为准。

相关项目与研究来源：

- OpenUnlearning：[项目](https://github.com/locuslab/open-unlearning)、[技术报告](https://arxiv.org/abs/2506.12618)。
- WMDP 与 RMU：[官方项目](https://github.com/centerforaisafety/wmdp)、[论文](https://arxiv.org/abs/2403.03218)。
- 选择题评测工具：[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- 起始模型：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)。

使用本仓库产出研究报告时，请同时引用实际使用的框架、方法、基准及模型；OpenUnlearning 的引用信息见上游 README 的 “Citing this work” 部分。

## 首轮正式实验（80步）

已完成原始模型基线、Bio语料清洗与分组划分、真实CUDA三步短跑、正式80步RMU训练、完整权重上传和训练后全量评测。模型使用BF16/SDPA；只更新第5/6/7层MLP down_proj权重，保存重载参数指纹一致。

| 0-shot准确率 | 训练前 | 训练后 | 变化 |
| --- | ---: | ---: | ---: |
| WMDP-Bio（1273题） | 73.8413% | 59.9372% | -13.9042个百分点 |
| MMLU（57学科，14042题） | 70.6737% | 65.8952% | -4.7785个百分点 |

训练后全部题目和评分请求与基线逐条一致；无聊天模板、无额外system instruction，batch1，选择题loglikelihood评分。WMDP下降表明代理任务目标能力减弱，同时应关注MMLU正常能力损失；不能宣称知识彻底消失，也不能将准确率直接作为安全总分。第80步在测试前固定，未按正式测试选择检查点。

评测证据与复现见 [训练后评测说明](docs/rmu-post-evaluation.md) 和 [配对汇总](logs/rmu_post_eval_summary.json)。原始基线及训练后逐题结果分别保存在 `open-unlearning/saves/eval/` 下，均保留。历史 `qwen3_4b_base_bio_smoke20` 因参数修复前limit未生效实际是全量，`smoke20_fixed` 才是20题验证。

## 数据、训练和模型留档

Bio按文章组划分，seed42，遗忘训练/验证23220/1217条，保留训练/验证57488/3029条，见 [数据处理记录](docs/data-preparation.md)。正式训练每类准备2048候选块及64独立验证块，256token、batch1、梯度累积16、80步。每类1280次输入，实际不同遗忘块1280、保留块920，并非遍历完整语料；方案及结果见 [正式训练记录](docs/rmu-formal-plan.md)。

正式模型完整权重、tokenizer、配置、许可证及加载说明位于 [正式训练目录](open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_formal80_v1/)，已通过Git LFS上传，完整性检查通过；训练/验证日志、实际块索引、来源指纹及保存重载检查均留档。原始模型文件清单见 [基模校验清单](logs/base_model_manifest.json)。

MMLU使用官方固定revision `c30699e8356da336a370243923dbaf21066bb9fe` 本地快照，按subject分组，不改写题目；0-shot结果不能直接与公开5-shot榜单比较。WMDP本次使用基线逐题记录重建的相同内容快照。

根目录私人工作文档不上传，公开README单独放行。独立环境、缓存、下载的原始基模及短跑完整权重留在本地。Bio遗忘语料条款限制未经许可向协作者名单之外分发；原始、清洗后正文和token缓存不公开提交，GitHub保存处理脚本、统计与来源记录。保留语料再分发许可尚待确认，平台接收方授权也需明确。

## 第一轮适配探索结果

完成实现审查与7组独立原始模型起点的40步训练，共21个开发检查点，并对2个候选进行了全量探索性评测。最佳已全量验证候选为 **hook12，更新10/11/12层down_proj.weight，alpha2，steering2，lr5e-5，第40步**；seed42、256token、batch1、累积16。

| 全量0-shot准确率 | 原始模型 | 首轮hook7/alpha1/80步 | 选定hook12/alpha2/40步 |
| --- | ---: | ---: | ---: |
| WMDP-Bio（1273题） | 73.8413% | 59.9372% | 66.3001% |
| MMLU（14042题） | 70.6737% | 65.8952% | 69.1355% |

选定候选Bio下降 **7.5412个百分点**，MMLU下降 **1.5382个百分点**，满足本轮MMLU下降不超过2点目标。它的遗忘幅度比首轮小、正常能力损失也明显更小；不把不同训练预算的改善全部归因于换层。

开发子集来自已观察过的官方test；第一阶段全量失败还参与了追加动机及候选排除，因此这些结果只称探索性，不是独立未参与调参的测试，也不证明知识彻底删除。第一阶段hook12/alpha1全量MMLU下降2.3074点未达目标，失败结果原样保留。学习率减半遗忘较弱，steering1第40步开发MMLU下降2.2222点，未通过初筛。

完整21个开发检查点、配置和两次全量对比见 [中文报告](docs/rmu-adaptation-results.md)、[CSV表](logs/rmu_adaptation_comparison_phase2.csv)、[最终结果](logs/rmu_adaptation_result_phase2.json)、[审查和预先声明方案](docs/rmu-adaptation-plan.md)。第一阶段报告单独保留于 [历史报告](docs/rmu-adaptation-results-phase1.md)。

全部新增检查点采用Git LFS管理的无损BF16 XOR+gzip增量，不是LoRA或独立完整HF模型；加载需固定原始模型及helper，重建逐位核查通过。最终模型指针和加载示例见 [适配产物说明](open-unlearning/saves/unlearn/rmu_adaptation_v1/README.md)，选定checkpoint为 `h12_a2_lr5e-05_s2_screen40/checkpoint-40`。数据盘已由50GB扩至60GB，本轮未使用/tmp缓存。

## 第二轮表示尺度与覆盖探索

完成表示与梯度诊断、全部既有开发结果的MMLU等权/题数加权重算、3组尺度/alpha实验及1组公平覆盖对照，并对预先锁定的2个候选进行全量评测。均从同一原始模型开始，hook12、更新10/11/12层down_proj、lr5e-5、seed42、40步。

| 全量探索性候选 | WMDP-Bio | MMLU | MMLU相对基线变化 |
| --- | ---: | ---: | ---: |
| 旧alpha2/steering2 | 66.3001% | 69.1355% | -1.5382点 |
| 新alpha2/steering32 | 55.3810% | 68.4874% | -2.1863点 |
| 新alpha1/steering32 | 50.6677% | 67.1913% | -3.4824点 |

新增全量候选均未达到原2点能力损失约束；该档仍保留旧alpha2/steering2。3点参考档的最佳已观测候选为新alpha2/steering32，比旧alpha2多降低Bio10.9191点、额外损失MMLU0.6481点。3点仅为本轮取舍参考，不代表通用遗忘成功。开发集两档最佳相同，但全量能力损失更大，体现小子集误差。

所有新模型与失败/负面结果保留。覆盖组严格匹配文章顺序和实际token预算，未见明确收益。10个新增增量实际约639.86MiB，保存重载逐位一致；第一组10/20步仅指标。数据盘60GiB，完成审计时余约9.12GiB，无需再次扩容。仅本地提交，未推送远端或发布模型。

见 [第二轮中文报告](docs/rmu-scale-v2-results.md)、[完整开发配置表](logs/rmu_scale_v2_development_config_comparison.csv)、[全量比较表](logs/rmu_scale_v2_full_comparison.csv)、[完成审计](logs/rmu_scale_v2_completion_audit.json)、[增量加载说明](open-unlearning/saves/unlearn/rmu_scale_v2/README.md)。所有全量结果仍为已观察官方test后的探索性结果，不证明知识彻底删除。多seed复验及恢复测试留作后续，本轮未自动开展。

## SimNPO独立对照与RMU联合探索

完成实现适配、mask/移位及CUDA梯度检查、3组各40步训练。固定同一原始模型、hook12、10/11/12层down_proj、lr5e-5、seed42、训练文章块和采样顺序；用独立文章验证梯度锁定系数，未按测试分数调参。

| 第40步开发结果（非全量） | Bio | 加权MMLU相对开发基线变化 |
| --- | ---: | ---: |
| 独立SimNPO＋retain CE | 73.8281% | -3.2403点 |
| RMU＋额外retain CE | 66.7969% | -2.1823点 |
| RMU＋SimNPO联合＋相同保留项 | 60.5469% | -4.1466点 |

开发基线Bio75.78125%，MMLU等权73.50877%、加权69.98410%，不能与全量基线混用。联合比同保留项RMU多降低Bio6.25点，但多损失加权MMLU1.9643点；所有新增开发点被既有候选支配，按预先规则未追加全量评测，不建议替换现有已验证RMU配置。单seed、三层限制和有限预算的负面结果不代表方法普遍无效。

9个无损增量保存重载校验通过，配置、分项loss、验证、时间、显存及逐题记录留档。见 [中文报告](docs/simnpo-rmu-results.md)、[方案](docs/simnpo-rmu-plan.md)、[比较表](logs/simnpo_rmu_development.csv)、[完成审计](logs/simnpo_rmu_completion_audit.json)、[恢复说明](open-unlearning/saves/unlearn/simnpo_rmu_v1/README.md)。本轮仅本地提交，未推送。

## 联合方法梯度诊断及系数减半

完成36层down_proj只读梯度诊断：固定4个文章验证样本，在原始、RMU+CE40、联合20/40检查点分别计算各损失梯度。RMU/Sim遗忘梯度均值接近正交，未见稳定强反向冲突；选定三层的加权Sim/RMU梯度范数比均值从约0.50升至联合40步约20.44，显示明显尺度漂移。三层以外有大量梯度，但这不是扩大参数必然有效的证据。

按预先门控只追加1组Sim系数减半（0.15665→0.07833），其余模型起点、参数范围、数据、采样、学习率、seed42和40步不变。第40步开发Bio63.2813%、加权MMLU损失3.4003点；原联合为60.5469%/损失4.1466点。减半缓解能力损伤但遗忘变弱，仍超过3点参考范围，未产生新前沿，未新增全量。

3个新增检查点及全部诊断/负面结果保留，重建和采样预算校验通过；没有扩大训练参数、继续搜索或推送。见 [诊断与适配报告](docs/simnpo-rmu-gradient-results.md)、[预先门控](docs/simnpo-rmu-gradient-plan.md)、[梯度证据](logs/simnpo_rmu_gradient_diagnostics.json)、[单因素审计](logs/simnpo_rmu_half_completion_audit.json)。下一步若继续，更值得先研究训练中的梯度尺度平衡，再预先声明有限对照。

## 训练中梯度尺度限幅对照

完成一组同预算40步对照：每个训练microbatch仅对实际3个可训练参数测RMU/Sim梯度范数，Sim权重取原上界与0.5倍梯度比公式的较小值；detach系数，不改已有梯度累积，验证/评测不更新控制器。其余模型起点、层、数据、采样、学习率和保留项不变。

640个训练微批次逐条核验通过，600次触发限幅，限幅后的Sim/RMU比值均<=0.5。第40步开发Bio66.0156%、加权MMLU损失3.1090点；原联合60.5469%/4.1466点，减半63.2813%/3.4003点。限幅缓解损伤，但Bio降低幅度也减小，未形成新前沿；终点只略超3点，不能因此断言方法无效，仍需结合全部既有候选比较。

按预先规则未新增全量，旧原始数据/模型/结果保持，3个增量无损重建，未追加rho搜索或扩层。本轮仅本地提交，尚未推送。见 [限幅报告](docs/simnpo-rmu-cap-results.md)、[预先规则](docs/simnpo-rmu-cap-plan.md)、[控制器汇总](logs/simnpo_rmu_cap_controller_summary.json)、[完成审计](logs/simnpo_rmu_cap_completion_audit.json)。单微批次优化器前约束不保证累积/Adam/MMLU约束。

## 下一步

交付整理暂缓；后续需要时按Word要求整理标准交付包：九个必需元数据文件、模型产物、实际训练与验证数据、前后评测证据及日志；受限数据依授权交付，缺项如实注明。当前标准平台交付包尚未完成。恢复攻击、再学习、多随机种子和多方法对照属于后续扩展。

服务器路径复现时需要调整，环境见 [依赖快照](logs/environment.txt)。模型与数据许可分别以各自来源为准，不公开token或其他凭证。
