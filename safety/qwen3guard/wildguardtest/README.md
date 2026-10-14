# 固定 WildGuardTest 分类器评测

仅调用既有 Qwen3Guard 审核器，不调用原模型/RMU 生成回答。固定来源与筛选规则见 `config.json`；结果与限制见 `REPORT.md`。

```bash
envs/unlearning/bin/python -m safety.qwen3guard.wildguardtest.prepare
envs/unlearning/bin/python -m unittest safety.qwen3guard.wildguardtest.test_prepare safety.qwen3guard.wildguardtest.test_evaluate safety.qwen3guard.test_adapter
envs/unlearning/bin/python -m safety.qwen3guard.wildguardtest.evaluate
envs/unlearning/bin/python -m safety.qwen3guard.wildguardtest.summarize
```

需要当前 HF 账号已经取得官方数据集权限及既有 Guard 本地快照。准备脚本下载固定测试 Parquet，保存原始行映射、任务清单、冲突清单和 SHA-256。所有文本及模型原始审核输出只保存于 Git 忽略的 `cache/`。`results.json` 仅含公开元数据及汇总，不含逐条文本。

审核器与生成配置保持不变。每任务先做一次非测试正常文本预热，前 6 个合格测试样本作为流程试运行并直接复用；不因预测正确与否改模板或筛选样本。每次审核完成即以 SQLite 事务持久化结果，缓存键包含任务、内容哈希、固定模型 revision 及推理/模板/解析器配置指纹。脚本内容改变也会改变缓存命名空间。按相同命令续跑即可，已有成功及错误结果不重复审核。连续 5 次新审核错误会停止；应先调查原因，不应通过改标签、模板或策略消除错误。

输出审核调用 `moderate(task, request, response)`，必须包含数据集已有回答。输入按请求精确去重；输出按官方行计数。两种策略由同一次审核结果计算。所有正常解析结果均以原有策略决定拦截；错误/未知在系统口径下视为不放行，在分类口径下单列并排除，同时公开覆盖率。

有害为正类。F1 为有害类 F1，Macro-F1 为有害与正常两类 F1 的等权平均。Precision/Recall/FPR/FNR 分母为零时为 null；类 F1 分母为零时记 0。Recall/FPR 为 Wilson 95% 区间。分组仅使用官方 `adversarial`；输入若同请求混合组元数据不强制归组，整体指标仍保留。所有评测结果是固定英文测试集上的离线分类表现。

计时从每条模板渲染开始，含 tokenization、设备拷贝、GPU 同步生成、解码、解析及策略计算；不含模型加载。任务 wall time 含逐条 SQLite 与状态文件落盘，排除预热、模型加载和最终汇总；吞吐按合格行数除以任务各运行会话 wall time。P50/P95 基于各条审核内部检测耗时，线性分位数。若续跑，读取原缓存时间，不将缓存检索延迟当成检测延迟；汇总会话时间、实际新增审核次数均保留。峰值为任务内 PyTorch allocated/reserved（已包含常驻审核模型），并非 nvidia-smi 整卡峰值。长度记录原始字符数、渲染后输入 token 数与实际生成 token 数；禁止截断。

`error_review.jsonl` 在本地缓存目录内，按 seed42 为各任务/策略有效预测的 FP、FN 各抽取最多 10 条。它含原文和原始审核输出，禁止提交。公开报告只描述非操作性错误类型，不把小批抽查当成新的准确率估计。

提交前明确暂存本阶段公开文件，再运行 `envs/unlearning/bin/python -m safety.qwen3guard.wildguardtest.check_publication`：复算缓存对应指标，核对执行源文件指纹，并检查暂存范围、文件类型/大小、HF token 格式及测试原文。此检查不调用模型。
