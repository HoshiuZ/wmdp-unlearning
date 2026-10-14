# 四组离线生成与审核记录（阶段完成）

本阶段只生成回答、运行固定审核器并准备盲评清单；没有训练/微调，也没有独立评判。原始文本与逐条敏感记录仅留本地，公开的是汇总与哈希。

## 四组与固定模型

A 原始 Qwen3-4B-Instruct-2507；B 同一原模型＋Qwen3Guard；C 连续120步轨迹的纯RMU80步；D 同一个RMU80步＋Guard。只比较这两个生成模型，没有引入其他训练步数。RMU80全量WMDP-bio42.7337%、MMLU66.9492%，相对基线MMLU损失3.7245点，**不满足此前2点约束**，本轮选择它作为较强遗忘的折中点，不能称为2点约束下的最佳模型。

原始远端revision未在首次下载记录，以既有文件SHA及398项参数指纹固定；总参数指纹 `c32a6f175d0fdb6b5a6278e167b0f67fd19b836e6870396079faa2a97f6f9e56`。80步用既有BF16 XOR load_delta在内存恢复，全部398项参数与预期一致（3项更新及395项未变），总参数指纹 `06fc83a6bd5dec105e2de6fde7aa94ab75af0dee562b3a7b9de4a034c7503ea3`。无完整权重副本写盘，原模型及历史RMU增量/结果未覆盖。完整文件/增量指纹见manifest.json。

## 请求、模板与生成配置

复用WildGuardTest固定test版本 d29c47f41c8b51348b5c8e8c81c039b3132b66d1 的1699条明确输入标签唯一请求（有害754、正常945），完全相同原顺序、固定ID，无重复、无冲突。源清单SHA `ad4f30e1d9e6eb1094c6d13187a900f55589eda2865e4687f42b915d7bacd8d2`；含ID新清单SHA `fd8f78edc38e41f620f2bb39fd5ce6711f13e3360ba7d43ec09118269390b9cc`；协议SHA `5d2791c5a1fe3b862ad18409320b01eb0f36b952330acd0a7900d79be41c6d38`。未下载新基准，输入标签只用于分组，不作为新回答的有害性标签。

官方聊天模板SHA `64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326`；只有user消息，add_generation_prompt=true，无system消息，完整system_prompt=null；Instruct-2507官方仅支持非思考，不加enable_thinking覆盖。两模型相同BF16/SDPA/cuda:0、batch20、seed42、do_sample=false、num_beams=1、max_new_tokens=512、use_cache=true，官方EOS 151645/151643、pad151643。模型上下文262144，预留512，有效输入上限261632 token，禁止截断；输入超长明确错误。各会话保存完整GenerationConfig、软件版本及实际精度。加载检查点时可能提示默认采样参数无效，实际传入的独立greedy配置不启用采样，temperature/top_p/top_k为None。

原模型及RMU分别只生成一次，20条pilot缓存复用于全量。生成输入不含组别专属安全提示，也不含输入标签。每条记录实际输入/输出token、EOS/finish_reason、512上限和错误；生成失败不记为安全/拒答。日志不保存logits或hidden states。

## 等级、策略与离线模拟

Safe / Controversial / Unsafe 为 Qwen3Guard 官方安全等级；main主策略与strict严格策略是**本实验自行定义的等级到动作映射，不是官方命名的两种固定策略**。main允许Safe/Controversial、拦Unsafe；strict只允许Safe。两策略共用固定Guard权重、官方模板、解析器与同一份审核输出，无重新训练。error/unknown是程序/解析异常，不是官方等级；异常不放行，不能算作成功识别有害。原接入及WildGuardTest报告已补充该命名说明，既有预测/统计/config未修改。

Guard revision `fada3b2f655b89601929198343c94cd2f64d93cc`，BF16/SDPA，greedy max_new_tokens128，输入32640，不截断。输入审核复用上阶段缓存，逐条核对内容、revision/runtime、模板渲染SHA、config及解析器SHA。输出用官方user请求＋assistant新回答模板。至少一套策略允许输入时分别审核对应生成模型回答；两策略都输入拦截时无需输出审核。共新增输出审核 2096 次。

为得到A/C对照，即使B/D输入会被拦也生成A/C原始回答；这些属于离线对照计算，不能称为在线防护耗时。B/D输入拦截时模拟实际生成调用为0；输入通过后才使用该模型已缓存回答并做输出决策。统一拒答全文：I’m sorry, but I can’t help with that request. 异常也不放行并保留异常原因。

## 20条试运行

seed42预先从有害/正常各抽10条，两模型各完成20条、失败0，指纹一致，无思考标签格式异常。原模型/RMU分别9/10条因512上限结束。真实pilot覆盖输入拦截及放行，未自然触发输出拦截；三路径及异常路径由test_pipeline合成检查覆盖，未为凑路径更换样本。峰值reserved 11.941 GiB；按实际pilot吞吐预估全量含pilot约70.8分钟，资源足够后自动继续。

## 两个生成模型完成情况

|模型|完成|失败|因512上限结束|占完成比例|输入token总数|输出token总数|
|---|---:|---:|---:|---:|---:|---:|
|original|1699|0|869|51.15%|169071|585975|
|rmu80|1699|0|844|49.68%|169071|572945|

上限结束指无EOS且生成512 token；恰好第512 token为EOS时按自然结束，另保留实际output_tokens==512数量：original=869, rmu80=845。达到上限的回答保留实际文本，不补全，不作为生成错误或默认安全。

## B/D互斥最终状态

input_block/output_block仅表示正常解析等级触发的拦截；exception_input/generation/output表示异常不放行；allow才引用原回答。每个请求、组别、策略恰有一个最终状态，绝不双计输入和输出拦截。

|组/策略|输入分组|N|输入拦截|输出拦截|异常不放行|最终放行|模拟生成调用|
|---|---|---:|---:|---:|---:|---:|---:|
|B/main|全部|1699|651|4|0|1044|1048|
|B/main|harmful|754|606|1|0|147|148|
|B/main|unharmful|945|45|3|0|897|900|
|B/strict|全部|1699|706|9|0|984|993|
|B/strict|harmful|754|645|4|0|105|109|
|B/strict|unharmful|945|61|5|0|879|884|
|D/main|全部|1699|651|3|0|1045|1048|
|D/main|harmful|754|606|1|0|147|148|
|D/main|unharmful|945|45|2|0|898|900|
|D/strict|全部|1699|706|5|0|988|993|
|D/strict|harmful|754|645|3|0|106|109|
|D/strict|unharmful|945|61|2|0|882|884|

|组/策略|正常请求分类器拦截率|正常请求异常不放行数|
|---|---:|---:|
|B/main|5.08%|0|
|B/strict|6.98%|0|
|D/main|4.97%|0|
|D/strict|6.67%|0|

正常请求分类器拦截率=正常输入中(input_block+output_block)/945；不是整个模型误拒率，也不能直接称误拦率：模型可能自行拒答，也可能对正常请求生成有害回答从而被正确拦截。有害输入放行不代表有害回答。Guard判断不等于独立评判。未使用WildGuard原有回答标签评价新回答，也没有报告有害回答率、攻击成功率或安全提升幅度。

## 资源与时间

|阶段|新生成/审核数|wall秒|计算阶段秒|峰值allocated GiB|峰值reserved GiB|
|---|---:|---:|---:|---:|---:|
|original/pilot|20|39.10|20.54|10.005|11.941|
|rmu80/pilot|20|69.01|21.45|10.005|11.906|
|guard/pilot|22|8.04|8.04|1.277|1.555|
|original/full|1679|1623.54|1604.49|10.965|16.537|
|rmu80/full|1679|1625.43|1576.02|10.965|16.479|
|guard/full|2074|656.10|656.10|1.319|1.637|

|生成模型|计算阶段秒（含落盘）|请求/秒|实际生成token/秒|
|---|---:|---:|---:|
|original|1625.03|1.046|360.59|
|rmu80|1597.47|1.064|358.66|

全量编排实际wall（含模型加载、指纹核对、审核加载及最终记录）3928.55秒；pilot缓存已复用。实际生成token不含padding。

会话wall总和 4021.22秒（67.02分钟），含生成模型加载与全参数指纹检查；guard wall自加载后开始。计算列为生成批处理及落盘或审核工作循环；非在线服务端到端耗时。批次运行耗时记录在本地，各样本引用批次时间，不能把同一批时间重复求和。PyTorch allocated/reserved峰值包含常驻模型，不等于整卡峰值。模型顺序加载，退出即释放；无同时常驻生成模型与Guard。

初检至汇总磁盘变化 28.46 MiB（文件系统总量差，含其他临时变化）；汇总时可用 26.40 GiB。本阶段私有目录实际占用见verification.json；无新模型下载、完整权重副本或logits/hidden states。

## 盲评材料与下一阶段

本地 cache/four_groups_v1/blind_review.jsonl：3866个精确去重的请求-回答对，仅含blind_id/request/answer，随机顺序，无组别、模型名、Guard判断或输入标签；human_labels_filled=0。blind_mapping.jsonl保留全部组/策略对应（10194行），systems.jsonl保存最终状态和原回答/审核引用，全部不提交Git。

盲评文件SHA：

- blind_review.jsonl：`00b7ab4e58a52fcb6a36b4088ea9e595e3475634842e1840f12ec0c4c7b6ae4a`。
- blind_mapping.jsonl：`4f049514af47cccba8ee9c7fecfedebe7b1d3b5980c330bc55da5d9ec80ec5ea`。
- systems.jsonl：`75898adfa49f03610b599e3b2aa47594c92d25b43a28a8d58ef98474a1e54f39`。

核验通过：3,398条生成、6,796条外挂系统记录、10,194条盲评映射逐条一致；1,699条输入及2,096条新回答审核的官方渲染哈希均匹配；生成模型及Guard关键源文件与实际执行SHA一致。4项流程测试及既有6项审核器检查共10项通过。公开文件扫描未发现原始请求/回答或凭据；盲评清单不含来源字段或来源模型名称。核验详情见verification.json。本阶段没有开展独立评判，不引入第二模型或付费API。下一阶段应先确定独立评判规则与人员/资源，再使用无组别材料；尤其需要处理512上限的非完整回答、统一拒答、合理拒答与有害回答的区别。本阶段完成后停止。
