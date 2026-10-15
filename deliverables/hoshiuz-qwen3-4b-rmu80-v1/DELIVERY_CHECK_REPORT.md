
# 交付包核验报告

核验时间：2026-10-15T04:49:51Z。本报告只核验本地元数据交付目录和既有证据；没有加载或重建模型，没有读取训练/评测语料正文，没有生成/评分回答。

**格式状态：** 必需字段、数组/对象/字符串/数值类型、源数据路径存在性、指标分母口径和文件索引结构已通过静态校验。**内容状态：部分待补。** 本包不是完整交付；具体null、未核实和未转交项目见 `MISSING_ITEMS.md`。

|检查项|结果|
|---|---|
|required_standard_files_present_before_manifest|True|
|required_standard_directories_have_readme|True|
|JSON_UTF8_and_finite_numbers|True|
|JSONL_UTF8_and_finite_numbers|True|
|timestamps_timezone_or_null|True|
|field_schema_validation|{'status': 'passed', 'dataset_rows': 8, 'capability_result_rows': 2, 'errors': []}|
|package_files_hashed_and_sized|True|
|package_files_record_count_field_and_values|True|
|fraction_percentage_denominator_consistency|True|
|credential_scan|{'pattern_hits': 0, 'hit_paths': []}|
|raw_text_scan|metadata builder reads no prompt/answer bodies or per-item assisted labels|
|percent_metric_range|{'percentages_0_to_100': True, 'risk_rates_0_to_1': True}|
|SVG_XML_parse|True|
|required_directory_readmes|True|
|model_rebuild_evidence_reused|True|
|model_rebuild_repeated_now|False|
|remote_LFS_verified|False|
|platform_submitted|False|
|content_complete|False|

RMU80重建证据复用现有 checkpoint-80/rebuild_verification.json：BF16 XOR压缩负载、基模张量和重建张量哈希匹配。检查点真实负载仍只在本机LFS缓存；Git远端可取回性未核实。该包的JSON/JSONL、字段类型、文件哈希、文件大小、record_count和SVG图表会在manifest.json生成后再次进行最终校验。

当前数据盘剩余 13814185984 bytes。此数值是核验时文件系统读数，不代表模型或语料已交付。
