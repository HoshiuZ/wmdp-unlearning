# WildGuardTest：Qwen3Guard 独立审核评测报告

日期：2026-10-14。已完成固定测试集、流程试运行及全部合格测试样本审核；本阶段到此停止。未训练模型，未调用原模型/RMU 生成回答，未执行正式四组实验。

## 数据来源与固定规则

官方 [WildGuard 仓库](https://github.com/allenai/wildguard) 指向 [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix)。数据集 `allenai/wildguardmix`，revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`，配置 `wildguardtest`，官方 `test` 文件 `test/wildguard_test.parquet`。许可 `odc-by`，另遵守 AI2 Responsible Use Guidelines 与访问申请条款。当前账号已获访问权限；此前 403 阻塞已解除。

实际 1725 行；唯一请求 1725，重复额外行 0，请求标签冲突 0。输入标签缺失 26，回答有害性缺失 16，拒答标签缺失 5。缺失未转为安全，三项标签独立筛选。

字段：prompt/response 为请求与数据集已有回答；两项 harm 标签为 harmful/unharmful，refusal 标签为 refusal/compliance；缺失可表示标注者未达成一致。agreement 为 2 或 3 名标注者一致（缺失为 null）。官方 adversarial 标记用于分组；subcategory 仅表示提示主题。实际 schema、分组与一致性分布见 data_metadata.json。

输入按原始请求精确去重并保留原始行号映射；冲突排除、不多数投票。只有一种已知标签且伴随缺失时沿用已知标签并保留映射，无已知标签则排除。输出按官方行计数，独立要求回答有害性标签有效及请求/回答非空；不要求输入标签有效。实际输出请求-回答对也无重复。

原始 Parquet SHA-256：`6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca`。任务清单 SHA-256：

- input：1699 条，`ad4f30e1d9e6eb1094c6d13187a900f55589eda2865e4687f42b915d7bacd8d2`。
- output：1709 条，`1be4206f8a6045a328fd62f6febbedca4f65b685ccfd016bf42f306733f64384`。
- conflicts：0 条，`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。

原始文件、原文、逐条审核输出、行映射及错误抽查仅在 Git 忽略的 cache/wildguardtest/ 中；公开产物不含原文。

## 固定审核方式与执行

模型 `Qwen/Qwen3Guard-Gen-0.6B`，revision `fada3b2f655b89601929198343c94cd2f64d93cc`，Apache-2.0；沿用已验证接入，模型、模板、解析器与策略未修改。BF16、CUDA、SDPA，greedy，do_sample=false、num_beams=1、max_new_tokens=128，输入上限 32640 token，禁止截断。输入仅 user 消息；输出为 user 请求＋assistant 的数据集已有回答；均由官方 tokenizer.apply_chat_template 渲染，无自拟系统词或附加生成后缀。

主策略只拦 Unsafe，严格策略拦 Unsafe 与 Controversial；同一审核结果计算两者。错误/未知不直接放行。每任务一次正常文本预热；原始顺序前 6 条试运行验证字段、任务及解析流程后自动继续，结果复用于全量。没有依据预测正确率改模板、策略或挑样本。逐条 SQLite 事务持久化；缓存键含任务、内容、模型 revision 及完整审核配置/模板/解析器指纹，连续 5 次错误停止。

评测启动时提交 `a66cefdf3eb453a6280f8b6371397c10714f3340`；新增执行代码的精确 SHA-256 及运行指纹见 results.json/provenance，最终提交包含相同执行文件。Python 3.11.17，PyTorch 2.9.1+cu128，Transformers 5.5.4；seed42。命令及续跑方式见 README.md。

## 核心结果

有害为正类。分类口径仅含有效解析样本；系统口径覆盖全部合格样本，error/unknown 视为不放行。F1 为有害类，Macro-F1 等权平均有害与正常两类。

|任务|样本数|有害/正常|错误/未知|覆盖率|
|---|---:|---:|---:|---:|
|input|1699|754/945|0 (0.00%)|100.00%|
|output|1709|284/1425|0 (0.00%)|100.00%|

|任务|口径|策略|N|TP/FP/TN/FN|Precision|Recall|F1|Macro-F1|FPR|FNR|
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
|input|classification_valid|main|1699|606/45/900/148|93.09%|80.37%|86.26%|88.29%|4.76%|19.63%|
|input|system_all|main|1699|606/45/900/148|93.09%|80.37%|86.26%|88.29%|4.76%|19.63%|
|input|classification_valid|strict|1699|645/61/884/109|91.36%|85.54%|88.36%|89.79%|6.46%|14.46%|
|input|system_all|strict|1699|645/61/884/109|91.36%|85.54%|88.36%|89.79%|6.46%|14.46%|
|output|classification_valid|main|1709|213/64/1361/71|76.90%|75.00%|75.94%|85.61%|4.49%|25.00%|
|output|system_all|main|1709|213/64/1361/71|76.90%|75.00%|75.94%|85.61%|4.49%|25.00%|
|output|classification_valid|strict|1709|233/89/1336/51|72.36%|82.04%|76.90%|85.96%|6.25%|17.96%|
|output|system_all|strict|1709|233/89/1336/51|72.36%|82.04%|76.90%|85.96%|6.25%|17.96%|

Recall/FPR 的 95% 置信区间使用 Wilson score（z=1.959963984540054），分母分别为实际有害与实际正常数；仅表达该固定测试集样本层面的二项不确定性，不包含领域转移或数据构建偏差。

|任务|口径|策略|Recall 95% CI|FPR 95% CI|
|---|---|---|---|---|
|input|classification_valid|main|[77.39%, 83.05%]|[3.58%, 6.31%]|
|input|system_all|main|[77.39%, 83.05%]|[3.58%, 6.31%]|
|input|classification_valid|strict|[82.85%, 87.87%]|[5.06%, 8.20%]|
|input|system_all|strict|[82.85%, 87.87%]|[5.06%, 8.20%]|
|output|classification_valid|main|[69.65%, 79.68%]|[3.53%, 5.69%]|
|output|system_all|main|[69.65%, 79.68%]|[3.53%, 5.69%]|
|output|classification_valid|strict|[77.16%, 86.07%]|[5.10%, 7.62%]|
|output|system_all|strict|[77.16%, 86.07%]|[5.10%, 7.62%]|

分母为零的 Precision/Recall/FPR/FNR 记 null；类 F1 分母为零时记 0。详细分组混淆矩阵和错误代码见 results.json。风险类别与拒答识别不作为主指标。

## 官方普通/越狱分组（系统口径）

|任务|adversarial|策略|N|Recall|FPR|Macro-F1|
|---|---|---|---:|---:|---:|---:|
|input|adversarial_False|main|903|83.05%|1.63%|91.15%|
|input|adversarial_False|strict|903|90.31%|3.47%|93.61%|
|input|adversarial_True|main|796|77.13%|8.13%|84.98%|
|input|adversarial_True|strict|796|79.77%|9.67%|85.36%|
|output|adversarial_False|main|912|79.74%|1.98%|90.57%|
|output|adversarial_False|strict|912|84.31%|2.50%|91.45%|
|output|adversarial_True|main|797|69.47%|7.36%|80.22%|
|output|adversarial_True|strict|797|79.39%|10.51%|80.34%|

## 错误抽查

固定 seed42，对各任务/策略有效解析结果的误拦 FP、漏检 FN 各抽取最多 10 条（同一例可出现在两策略中），保存原文、原始审核输出及官方标签/主题/拒答信息到本地 error_analysis.jsonl；公开 error_summary.json 仅含元数据汇总。主题指请求主题，不能解释为回答风险类别。逐条保留标签及元数据核对，并对代表样例补充本地语义复核注释；此抽查不产生新的准确率估计。

- input/main/FP：10 条；预测严重度 {'Unsafe': 10}；官方回答拒答标签 {'compliance': 10}。
- input/main/FN：10 条；预测严重度 {'Safe': 9, 'Controversial': 1}；官方回答拒答标签 {'refusal': 5, 'compliance': 5}。
- input/strict/FP：10 条；预测严重度 {'Controversial': 2, 'Unsafe': 8}；官方回答拒答标签 {'compliance': 9, 'refusal': 1}。
- input/strict/FN：10 条；预测严重度 {'Safe': 10}；官方回答拒答标签 {'refusal': 8, 'compliance': 2}。
- output/main/FP：10 条；预测严重度 {'Unsafe': 10}；官方回答拒答标签 {'compliance': 9, 'refusal': 1}。
- output/main/FN：10 条；预测严重度 {'Safe': 10}；官方回答拒答标签 {'compliance': 10}。
- output/strict/FP：10 条；预测严重度 {'Controversial': 3, 'Unsafe': 7}；官方回答拒答标签 {'compliance': 8, 'refusal': 2}。
- output/strict/FN：10 条；预测严重度 {'Safe': 10}；官方回答拒答标签 {'compliance': 9, 'refusal': 1}。

抽查的主要误差类型：输入误拦集中于官方标正常、但涉及隐私/生物特征、政治传播或创作角色包装的边界请求；输入漏检既含角色包装、偏见前提等风险，也包含表面合规、防护或虚构语境。后者提示官方二分类标签与分类器风险定义可能存在差异，不能据此自行重标或剔除。

输出误拦涉及创作对话、引用分析、版权与专业建议边界，也出现实际拒答/反歧视论述仍被拦的例子。输出漏检包括角色包装的隐私侵犯、偏见表述与医疗建议，以及以拒答或安全声明开头、但官方仍标为有害且 compliance 的回答。只看回答开头或 Refusal 字段不足以判断全文安全。这里只给非操作性概述，逐条文本与复核注释留在本地。

## 时间、长度与资源

检测延迟含模板渲染、分词、设备拷贝、同步生成、解码、解析及策略计算；排除模型加载、预热和写盘。任务 wall time 另含逐条持久化与状态文件写入；吞吐按合格行数/wall time。P50/P95 为每条检测耗时的线性分位数。它们均不是完整防护服务的端到端延迟。

|任务|wall 秒|吞吐 条/秒|平均检测 ms|P50 ms|P95 ms|峰值 allocated GiB|峰值 reserved GiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|input|375.54|4.52|217.49|191.54|290.10|1.239|1.637|
|output|557.35|3.07|321.02|305.31|406.23|1.666|1.822|

PyTorch 峰值包含常驻 Guard 参数，allocated/reserved 分别列出；不等于整卡占用峰值。每次会话实际新增审核数和时间均保存；原有缓存结果不重复推理。

|任务|长度类型|P50|P95|最大|
|---|---|---:|---:|---:|
|input|request_characters|250.0|1440.6|3013|
|input|response_characters|0.0|0.0|0|
|input|input_tokens|342.0|567.1|908|
|input|generated_tokens|8.0|12.0|14|
|output|request_characters|240.0|1438.8|3013|
|output|response_characters|1639.0|3964.8|22038|
|output|input_tokens|789.0|1315.0|4141|
|output|generated_tokens|13.0|17.0|19|

测试 Parquet 下载 2,258,878 字节（约 2.15 MiB），复用 HF 缓存，无训练集下载。Guard 权重沿用上一阶段单份缓存，约 1.415 GiB（含必要 tokenizer/config 文件），本阶段无新增模型权重下载。本阶段本地任务清单、缓存及抽查记录占用约 9.27 MiB（不含共享 HF Parquet）；汇总时数据盘可用 26.43 GiB。

## 下一阶段与限制

以上是 WildGuardTest 英文请求和数据集已有回答上的分类器离线表现，不能直接推断原模型/RMU 加防护后的安全提升、拒答率或服务延迟。下一阶段仍需固定生成样本、回答生成参数及四组一致的评测流程，并明确 Controversial 与 WildGuard 二分类标签之间的定义差异；本阶段不启动这些工作。数据准备、指标及既有接入共 11 项检查通过；3,408 条持久化审核记录与公开系统指标逐条复算一致，执行源文件指纹核对一致，暂存内容未发现测试原文、HF 凭据或大文件。本轮未发现模板、标签映射或解析流程的实现错误；所有结果均符合固定配置，失败数为零。公开测试集与分类器训练语料是否重叠不在本阶段核实范围。正式四组对照建议仍以预先固定的 Unsafe-only 为主结果，同时报告严格策略：主策略在两任务的正常样本误拦率较低，便于同时考察防护与过度拦截；严格策略提供更高召回，应作为明确注明的敏感性对照，而非事后替换主策略。策略配置没有修改。

## 策略命名说明（仅补充说明，不改变历史结果）

Safe / Controversial / Unsafe 是官方输出等级；“主策略/main”和“严格策略/strict”是本实验自行定义的等级到动作映射，不是官方命名的两种固定策略。主策略放行 Safe/Controversial、拦 Unsafe；严格策略只放行 Safe。error/unknown 是执行或解析异常，不是官方等级，异常不放行并单列，不计为成功识别有害。两种策略共用模型、模板与审核结果，不涉及重新训练。
