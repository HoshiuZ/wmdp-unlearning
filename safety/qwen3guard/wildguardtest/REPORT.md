# WildGuardTest 独立评测：准备完成，数据权限阻塞

日期：2026-10-14。当前状态：**未运行试审核或全量审核，不能报告分类成绩。**

## 已核实来源

官方 [WildGuard 仓库](https://github.com/allenai/wildguard) 与 [WildGuardMix 数据集卡](https://huggingface.co/datasets/allenai/wildguardmix)；数据集 `allenai/wildguardmix`，固定 revision `d29c47f41c8b51348b5c8e8c81c039b3132b66d1`，配置 `wildguardtest`，官方 `test` 划分，文件 `test/wildguard_test.parquet`。许可标识 `odc-by`，另需接受 AI2 Responsible Use Guidelines、填写个人/机构信息并同意信息使用条款。

数据集卡称测试集 1,725 行；这是来源声明，尚未下载实际文件核验。`prompt_harm_label`、`response_harm_label` 为 harmful/unharmful/缺失；缺失可表示标注者未达成一致。`response_refusal_label` 为 refusal/compliance/缺失。三个任务的标签可独立缺失。官方另有各任务 agreement 字段、`adversarial` 布尔分组、提示风险 `subcategory`。尚未实际检查 Parquet schema 或标签分布。

## 实际阻塞与已完成检查

已使用服务器当前 Hugging Face 凭据访问上述固定 revision 的正确测试文件，返回 `GatedRepoError / HTTP 403`，提示账号不在授权列表。公开 README 可以读取，不代表测试数据权限已获批。未尝试非官方镜像、绕过门禁或代填隐私/机构信息。

请使用服务器当前 token 对应账号，在 [官方数据集页面](https://huggingface.co/datasets/allenai/wildguardmix) 完成访问申请与条款确认。若换账号，需要更新服务器凭据；不要把 token 发到聊天或提交 Git。

本阶段完成固定来源、筛选规则与本地数据准备脚本，尚未实现/运行完整推理调度、指标汇总与错误抽查流程。新增数据准备测试 2 项，加原审核接入测试 6 项，共 8 项通过。这些是合成数据/接口检查，不能代替真实字段与小样本验证。

## 固定的数据规则

配置见 `config.json`。仅下载固定测试文件；不下载训练集。原始文件复用既有 HF 缓存；清单、逐条文本、行映射写入 Git 忽略的 `cache/wildguardtest/<revision>/`。

输入按原始请求字符串精确去重，不改变大小写或空白。保留原始行号映射；同一请求出现两种已知标签则排除并单独记录，不多数投票。若只有一种已知标签，允许沿用该标签，同时保留缺失标签对应行映射；完全没有已知标签则排除。输出按官方行保留，只要求请求、已有回答非空及回答有害性标签有效，不依赖输入标签有效。意外标签直接报错，不转换为安全。

准备脚本保存实际行数、重复数、冲突数、缺失数、标签分布及原始文件/各清单 SHA-256；当前因权限阻塞这些统计尚未生成。

```
envs/unlearning/bin/python -m safety.qwen3guard.wildguardtest.prepare
envs/unlearning/bin/python -m unittest safety.qwen3guard.wildguardtest.test_prepare safety.qwen3guard.test_adapter
```

## 审核配置及后续停止条件

既有 Guard revision `fada3b2f655b89601929198343c94cd2f64d93cc`、官方输入/输出模板、解析器及 `safety/qwen3guard/config.json` 均未修改。主策略仍只拦 Unsafe，严格策略拦 Unsafe 与 Controversial；错误/未知不直接放行。

权限恢复后先核实实际 schema/标签，固定清单和完整推理配置，再对各任务原始顺序前 6 条做流程验证，并复用结果自动继续全量。续跑缓存需包含任务、内容哈希、模型 revision、模板/解析器与推理配置指纹；逐条持久化，连续 5 次失败停止；超长不截断。输出审核必须输入请求及已有回答。

需报告有效解析子集分类指标与全部合格样本 fail-closed 拦截指标，正类为有害；各含混淆矩阵、Precision/Recall/F1/Macro-F1/FPR/FNR，错误与覆盖率。Recall/FPR 使用 Wilson 95% 区间。固定 seed42 抽查错误并只在本地保存文本。完成后才可基于结果给出策略建议；本次没有证据更改或推荐替换预设主策略。

## 当前结果与资源

| 项目 | 当前值 |
|---|---|
| 输入/输出实际样本数及类别分布 | 未取得数据，未知 |
| 两策略 Recall / FPR / Macro-F1 | 未评测，不可计算 |
| error/unknown 比例 | 未评测，不可计算，不能写作 0% |
| 审核模型调用数 | 0 |
| 审核耗时/P50/P95/吞吐/峰值显存 | 未运行，不适用 |
| 测试数据下载量 | 0；仅取得公开元数据/README |
| 数据盘可用空间 | 约 27 GiB（df -h，提交前） |

未训练模型，未调用原模型/RMU 批量回答生成，未改动既有 RMU 产物，未新增模型下载。未运行正式四组实验。当前仅能提交可公开的准备代码、配置与阻塞报告；原始数据、缓存、凭据及逐条危险文本不提交。
