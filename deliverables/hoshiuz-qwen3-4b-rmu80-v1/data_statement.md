
# 数据说明

本交付包只包含数据来源、许可/访问状态、统计、哈希和本地路径索引；不包含训练语料、验证块、评测题、WildGuardTest请求/回答、四组生成回答或逐条辅助标签。

## 后训练数据

RMU训练输入来自本地CAIS WMDP-Bio forget论文语料和Bio retain语料。Forget源记录为 cais/wmdp-bio-forget-corpus，revision 5a786ed1041e56fef2d4ed26c1239f12b73a68eb；其官方数据卡限制未经许可不得向协作者名单之外分发。Retain源从归档WMDP数据包读取，原始来源revision和再分发许可没有记录，不能推断框架MIT许可证覆盖语料。交付接收方必须单独确认资格和授权。

原始forget/retain记录分别为 24453 / 60887；清洗后的训练/验证文章计数列在 training_datasets.jsonl。清洗版本 bio-clean-v1；交叉角色冲突及精确重复处理规则、近重复启发式、固定文章池和token块构造规则均保留在本文件及项目清单中。RMU80实际来自2048个固定候选块/角色，至80步分别有 1280 / 1280 次block exposure，unique为 1280 / 920。候选块数不是曝光数，曝光数也不是训练文章全文数。

存档记录显示对 15600 个评测问题做过受限的匹配筛查，确认匹配数为 0；方法不是语义/答案级全覆盖，不能写成无污染保证。MMLU和WMDP-Bio官方test已参与选择/调参，相关分数为探索性结果。

## 评测数据

WMDP-Bio test源为 cais/wmdp/wmdp-bio，本地快照1273条；远端revision复核标记为 False。WildGuardTest来自 allenai/wildguardmix revision d29c47f41c8b51348b5c8e8c81c039b3132b66d1，官方test split许可记录为 ODC-BY；本包不分发原文。MMLU来源 cais/mmlu revision c30699e8356da336a370243923dbaf21066bb9fe，存档卡记录MIT许可。

## 模型辅助复核与隐私

30条ChatGPT辅助标签属于模型辅助判断，具体模型版本未记录，全部human_verified=false；未完成人工金标准。交付包只含分组聚合和匿名统计，不含逐条标签、理由、原文或映射。空值保持不确定，未被当作安全。

## 分发状态

训练及评测数据当前只在项目服务器本地。包外绝对路径、Git LFS指针和代码仓库状态均不等于数据已经交给平台。不得把本包描述为自包含数据交付。
