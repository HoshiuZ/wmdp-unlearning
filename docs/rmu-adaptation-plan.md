# Qwen3-4B RMU 小范围适配：审查和预先固定方案

## 实现审查

首轮正式解析配置 `open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_formal80_v1/training_config.json` 与全量证据核实：hook7，更新5/6/7 down_proj.weight，80步、累积16、lr5e-5、alpha1、gamma1、steering2。Bio73.8413→59.9372%，MMLU70.6737→65.8952%。不能将三步短跑配置当成正式配置。

实际起始模型Qwen3-4B-Instruct-2507，Qwen3ForCausalLM，36层、hidden2560、intermediate9728，4022468096参数。远端immutable revision未在最初下载记录，版本由 `logs/base_model_manifest.json` 的本地SHA256及全部参数指纹固定，不虚构revision。模型配置的旧transformers_version是来源元信息，实际运行版本是5.5.4。

层号从0开始：model.layers.7是第8个decoder block；forward hook捕获整个block的输出（tuple则首项），不是输入、不是MLP单独输出。每组更新3个MLP down_proj.weight，各2560×9728，共74711040参数；其他参数冻结且不加入优化器。最小CUDA检查验证每个选中参数梯度有限且非零、确实更新，其余及参考模型参数不变。

参考模型是原模型deepcopy，eval且requires_grad=False，其表示在no_grad下计算。训练模型train，验证模型eval，验证后恢复train；Qwen attention_dropout=0。验证执行不改变训练随机数状态。MSE先在hidden维平均，每个样本按labels!=-100的有效token平均，再对batch平均；总损失gamma×forget+alpha×retain，retain是同hook表示对冻结参考模型的MSE。首次token标签固定-100，也排除表示损失；这属于当前既有协议。padding labels=-100，全部实际token缓存均无原生pad token151643。

控制向量首次由CPU torch.rand均匀[0,1)生成，整体L2归一化后乘steering，shape1×1×2560，对全部遗忘token广播，并在同次运行固定。实际向量和哈希保存。不同层候选用相同seed与初始化顺序，核查向量一致。BF16转换后的范数允许舍入误差。默认方法签名steering20不等于当前解析配置steering2。

遗忘数据来自获批cais/wmdp-bio-forget-corpus固定revision5a786ed1041e56fef2d4ed26c1239f12b73a68eb，保留数据来自官方S3 Bio-retain。清洗、精确去重、文章组划分seed42：遗忘train23220/val1217，保留train57488/val3029。正式缓存每类2048训练候选文章块、64独立文章验证块，一篇一个至多256token块，无chat模板、无跨文档拼接，直接使用缓存IDs。哈希和具体规则见logs/rmu_formal_blocks_manifest.json；筛选继续同一缓存与seed，不将候选数冒充实际用量。

既有CPU梯度、batch掩码、梯度累积16、hook清理、验证no_grad/模式恢复及NLL加权检查通过。未发现必须修正既有训练行为的明确错误。仍以真实CUDA最小运行为准；如后续发现错误，先记录修正并重建对照。

## 固定的小范围预算与选择

统一40 optimizer steps，batch1、累积16、256token、lr5e-5、alpha1、steering2、gamma1、AdamW、恒定学习率、无warmup、seed/data_seed42。与首轮80步不同，属于本轮筛选预算；所有层候选预算相同。从同一个原始模型重新开始。检查点10/20/40，不只看最后一步。先单独做各层2步最小运行，不把短跑权重用于筛选训练。

第一阶段hook7/12/18对应更新5,6,7 / 10,11,12 / 16,17,18 down_proj.weight。第二阶段在第一阶段最佳层上比较alpha1/2/4，其中alpha1复用同预算结果，新增两组独立训练。按开发集MMLU相对同子集基线下降<=2个百分点过滤，再最小化Bio，平手优先MMLU更高，再优先较少步数。若无满足者，则选择MMLU损失最小层做alpha比较并如实记录；若仍没有满足者，最多各追加一个单因素候选lr2.5e-5或steering1，条件不同时修改，不做大规模搜索。

开发集固定哈希选择Bio256题、MMLU每学科30题共1710题；MMLU开发平均是每学科等权近似筛选，全量仍14042题按样本加权。与原始基线完全相同的逐题提示和四个评分请求，0-shot、无system/chat、BF16/SDPA、batch1、argmax loglikelihood。开发子集baseline来自已保存原模型逐题记录，并实测验证直接请求评分与原基线一致。

没有独立WMDP-Bio验证题集：开发集来自已全量评测过的官方test，首轮全量结果亦已影响本轮设计。因此所有本轮最终全量结果标为探索性，不称独立未参与调参的测试；其余题未用于本轮候选排序但此前全量统计已被观察。训练中的独立retain文章块只监控语料NLL/表示损失，不能替代MMLU验证。子集2点约束有抽样不确定性，最终全量确认阈值为MMLU>=68.673693%。全量仅评一个已锁定候选，若失败如实报告，不继续按全量反复挑选。

## 资源与检查点

4090D24GB，首轮峰值18.24GiB。本轮先检查较深hook能否运行。数据盘仅约1.1GB，不复制完整模型，不删除已有模型。新检查点保存三个BF16参数与原始参数uint16逐位XOR，经gzip无损压缩；记录形状、基模/训练后tensor hash及文件SHA，重建必须逐位一致。权重archive由Git LFS缓存保留，工作树可仅保留LFS pointer；加载脚本可从本地LFS对象读取。该自定义格式不是PEFT/LoRA，复现需原始固定基模及scripts/rmu_adaptation_utils.py。优化器状态不保存，模型可重建但不是断点续训包。训练配置、逐步指标、耗时显存、采样indices、验证和开发逐题评分保存独立目录，不覆盖历史。
