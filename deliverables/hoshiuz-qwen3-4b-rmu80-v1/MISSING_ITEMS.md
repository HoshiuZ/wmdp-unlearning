
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

## 元数据内容待补或待确认

格式字段已经按当前交付规范整理；下列内容仍缺失或待接收方确认，不能据此宣称包内容完整。

|字段|状态|原因|
|---|---|---|
|`model_card.base_model_version / model_artifacts.base_model.remote_revision`|unknown|基模最初下载时未记录上游不可变revision；目前以本地文件清单和参数指纹固定版本。|
|`training_datasets[].raw_count for cleaned article splits`|null_not_recorded|归档只记录forget/retain各自清洗前的合计数，未记录按train/validation拆分的清洗前数量。因此分割级raw_count为null，另列角色合计数和清洗后分割数。|
|`training_datasets[].license_or_authorization for retain data`|unresolved|retain语料归档revision及接收方再分发授权尚未从已有记录中确认。|
|`model_card.artifact_access.remote_retrievable`|null_unverified|未核实远端Git LFS对象可取回；本机缓存存在不代表远端可用。|
|`post_training.platform_method_enum / model_card.model_type enum mapping`|pending_recipient_platform|实际方法记录为RMU；machine_unlearning是本项目扩展表达，目标平台枚举尚未提供。|
|`manifest.submitted_at`|null_not_submitted|本阶段未向平台提交；实际转交前必须保持null。|

RMU40/80/120增量仍在本机LFS缓存；本轮记录不证明GitHub端已可取回。数据权限、平台入口和服务部署均未由本阶段完成。
