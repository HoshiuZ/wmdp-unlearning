
# 缺项与接收前事项

当前状态是元数据与报告已准备，不是完整模型/数据已经交到平台。下表逐项区分本地状态、远端状态和后续条件。

|项目|状态|后续事项|
|---|---|---|
|完整Qwen3-4B原始基模权重及配置|present_locally_not_transferred|obtain authorized exact base model; official immutable source revision was not recorded|
|RMU40/80/120 BF16 XOR+gzip增量|present_in_local_LFS_cache_remote_unverified_not_transferred|verify/upload/retrieve all required LFS objects or provide another authorized transfer|
|原始tokenizer文件|present_locally_not_transferred|transfer with exact base model or provide authorized official access|
|WMDP-Bio forget/retain训练语料与固定token块|present_locally_restricted_not_transferred|verify recipient collaborator authorization; retain-corpus redistribution rights are unresolved|
|WMDP-Bio/MMLU/WildGuardTest评测原文|present_locally_or_source_retrievable_not_transferred|verify applicable source access/license and retrieve from official source|
|在线推理服务及运维材料|missing|out of scope for this phase; no service was built|
|平台模型遗忘方法枚举映射|missing_or_not_found|platform must define a machine-unlearning/RMU method mapping; do not label as SFT or DPO|
|人工安全复核金标准|not_completed|ChatGPT-assisted labels remain auxiliary and human_verified=false|
|独立未参与调参的最终测试|not_available|official tests were used in tuning; current results remain exploratory|
|PDF版中文报告|not_created|Markdown report is complete; no reliable Chinese PDF export was verified and no heavy toolchain was installed|
|平台实际提交时间|not_submitted|manifest.submitted_at remains null until an actual platform submission|

RMU40/80/120增量仍在本机LFS缓存；本轮记录不证明GitHub端已可取回。数据权限、平台入口和服务部署均未由本阶段完成。
