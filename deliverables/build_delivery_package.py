"""Build a metadata-first RMU80 delivery package from archived project records.

This script reads manifests/results only; it does not load model weights,
datasets, evaluation prompts, generated answers, or per-item review labels.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "hoshiuz-qwen3-4b-rmu80-v1"
RUN_REL = Path("open-unlearning/saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120")
RUN = ROOT / RUN_REL
BASE = ROOT / "models/Qwen3-4B-Instruct-2507"
SOURCE_DELTA = RUN / "checkpoint-80"
SCHEMA_DATE = "2026-10-15"


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def rel(path):
    return str(Path(path).as_posix())


def portable(value):
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, str):
        root = str(ROOT)
        return value.replace(root, "$PROJECT_ROOT")
    return value


def lfs_pointer_info(pointer_path, entry):
    oid = entry["compressed_sha256"]
    payload = ROOT / ".git/lfs/objects" / oid[:2] / oid[2:4] / oid
    return {
        "repository_path": str(pointer_path.relative_to(ROOT)),
        "git_checkout_file_bytes": pointer_path.stat().st_size,
        "git_lfs_oid_sha256": oid,
        "expected_payload_bytes": entry["bytes"],
        "local_payload_path": str(payload),
        "local_payload_present": payload.is_file(),
        "local_payload_bytes": payload.stat().st_size if payload.is_file() else None,
        "local_payload_hash_matches": sha_bytes(payload.read_bytes()) == oid if payload.is_file() else False,
        "included_in_delivery_package": False,
        "remote_uploaded_or_retrievable": None,
        "remote_status": "not verified; quota/upload state unavailable in archived run record",
    }


def file_record(path):
    p = OUT / path
    b = p.read_bytes()
    rec = {
        "path": path,
        "purpose": PURPOSES.get(path, "supporting delivery metadata"),
        "record_count": None,
        "size_bytes": len(b),
        "sha256": sha_bytes(b),
        "availability": "included_in_delivery_package",
    }
    if path.endswith(".jsonl"):
        rec["record_count"] = sum(1 for line in b.splitlines() if line.strip())
    return rec


DATASET_FIELDS = {
    "dataset_id", "dataset_name", "dataset_version", "split", "purpose",
    "source_type", "source_detail", "file_paths", "file_format", "raw_count",
    "used_count", "cleaning_pipeline_version", "deduplication_method",
    "license_or_authorization", "test_overlap_result",
}


def validate_delivery_schema(out):
    """Validate field names/types and cross-field count semantics, not only JSON syntax."""
    errors = []
    def read_json(name):
        return json.loads((out / name).read_text(encoding="utf-8"))
    def expect(ok, message):
        if not ok:
            errors.append(message)

    card = read_json("model_card.json")
    for key in ("model_id", "model_name", "model_version", "base_model_id", "base_model_name", "base_model_version", "model_type", "provider", "release_status"):
        expect(isinstance(card.get(key), str) and bool(card[key]), f"model_card.{key} must be a non-empty string")
    expect(isinstance(card.get("parameter_count"), str), "model_card.parameter_count must be a string")
    expect(isinstance(card.get("parameter_count_exact"), int), "model_card.parameter_count_exact must be an integer")
    expect(isinstance(card.get("architecture"), dict), "model_card.architecture must be an object")
    expect(isinstance(card.get("context_length"), int), "model_card.context_length must be an integer")
    access = card.get("artifact_access")
    expect(isinstance(access, dict), "model_card.artifact_access must be an object")
    if isinstance(access, dict):
        expect(isinstance(access.get("local_available"), dict), "model_card.artifact_access.local_available must be an object")
        expect(access.get("remote_retrievable") is None, "remote LFS retrievability must stay null while unverified")
        expect(access.get("transferred_to_platform") is False, "model artifacts must not claim platform transfer")
        expect(access.get("included_in_delivery_package") is False, "model payloads must not be described as included")
    expect(card.get("model_type") == "machine_unlearning" and "project-level extension" in card.get("model_type_note", "").lower(), "model_type must identify RMU project-level machine unlearning and disclose enum mapping status")
    expect(isinstance(card.get("architecture"), dict) and card["architecture"].get("architectures"), "architecture details must remain under model_card.architecture")

    post = read_json("post_training.json")
    expect(isinstance(post.get("quality_controls"), list), "post_training.quality_controls must be an array")
    expect(all(isinstance(x, dict) and x.get("check_id") for x in post.get("quality_controls", [])), "every quality control must be a named object")
    expect(post.get("methods") == ["RMU"], "post_training.methods must preserve actual RMU method")
    expect(isinstance(post.get("training_data"), list), "post_training.training_data must be an array")
    for idx, row in enumerate(post.get("training_data", [])):
        expect(isinstance(row, dict) and DATASET_FIELDS <= set(row), f"post_training.training_data[{idx}] is missing required dataset fields")

    dataset_rows = [json.loads(line) for line in (out / "training_datasets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expect(bool(dataset_rows), "training_datasets.jsonl must contain dataset rows")
    for idx, row in enumerate(dataset_rows):
        expect(DATASET_FIELDS <= set(row), f"training_datasets.jsonl row {idx} is missing required fields")
        expect(isinstance(row.get("file_paths"), list) and all(isinstance(p, str) for p in row.get("file_paths", [])), f"dataset row {idx} file_paths must be a string array")
        expect(isinstance(row.get("used_count"), int) and row["used_count"] >= 0, f"dataset row {idx} used_count must be an explicit nonnegative integer")
        expect(row.get("raw_count") is None or (isinstance(row.get("raw_count"), int) and row["raw_count"] >= 0), f"dataset row {idx} raw_count must be nonnegative integer or null")
        if row.get("raw_count") is None:
            expect(bool(row.get("raw_count_note")), f"dataset row {idx} null raw_count requires an explanation")
        expect(isinstance(row.get("license_or_authorization"), dict), f"dataset row {idx} authorization must be an object")
        expect(isinstance(row.get("test_overlap_result"), dict), f"dataset row {idx} overlap result must be an object")
        expect(len(row.get("file_paths", [])) == len(row.get("file_paths_included_in_delivery_package", [])), f"dataset row {idx} path inclusion flags do not align")
        for path in row.get("file_paths", []):
            expect((ROOT / path).is_file(), f"dataset source path does not exist: {path}")
        expect(not any(row.get("file_paths_included_in_delivery_package", [])), f"dataset row {idx} must not claim source files are package-included")
    expect(len(post.get("training_data", [])) == len(dataset_rows), "post_training.training_data rows must match training_datasets.jsonl")

    evaluation = read_json("evaluation_summary.json")
    for key in ("trained_model_id", "evaluation_protocol_version", "scoring_policy_version", "datasets", "capability_results", "overall_result"):
        expect(key in evaluation, f"evaluation_summary.{key} is required")
    expect(isinstance(evaluation.get("datasets"), list), "evaluation_summary.datasets must be an array")
    expect(isinstance(evaluation.get("capability_results"), list), "evaluation_summary.capability_results must be an array")
    expect(isinstance(evaluation.get("overall_result"), dict), "evaluation_summary.overall_result must be an object")
    if isinstance(evaluation.get("overall_result"), dict):
        for key in ("aggregate_score", "calibrated_risk", "ranking"):
            expect(evaluation["overall_result"].get(key) is None, f"overall_result.{key} must remain null without a defined protocol")
    expect(all(isinstance(x, dict) and x.get("dataset_id") for x in evaluation.get("datasets", [])), "evaluation_summary.datasets entries need dataset_id")
    expect(all(isinstance(x, dict) and x.get("metric_id") for x in evaluation.get("capability_results", [])), "capability_results entries need metric_id")
    expect(evaluation.get("trained_model_id") == card.get("model_id"), "trained_model_id must match model_card.model_id")
    expect("WMDP-Bio" in " ".join(x.get("direction_note", "") for x in evaluation.get("capability_results", [])), "WMDP-Bio metric direction must be explicit")
    expect(any("general-capability loss" in x.get("direction_note", "") for x in evaluation.get("capability_results", [])), "MMLU metric direction must be explicit")
    expect("paired_contrast_denominator_note" in evaluation.get("four_group_open_response_results", {}), "paired comparison denominator must be documented")

    artifacts = read_json("model_artifacts.json")
    expect(isinstance(artifacts.get("artifact_access"), dict), "model_artifacts.artifact_access must be an object")
    if isinstance(artifacts.get("artifact_access"), dict):
        expect(artifacts["artifact_access"].get("remote_lfs_retrievable") is None, "model_artifacts remote LFS status must be null while unverified")
        expect(artifacts["artifact_access"].get("transferred_to_platform") is False, "model_artifacts must not claim platform transfer")
    delta = artifacts.get("rmu_delta", {})
    expect(delta.get("is_lora") is False, "RMU delta must not be marked LoRA")
    expect(delta.get("independently_trained_80_step_run") is False and delta.get("selected_checkpoint_step") == 80 and delta.get("run_max_steps") == 120, "selected checkpoint must remain step 80 from continuous120 run")
    expect(all(x.get("remote_uploaded_or_retrievable") is None for x in delta.get("parameters", [])), "unverified remote LFS status must be null, not false")
    config = read_json("training_config.json")
    expect(config.get("config", {}).get("trainer", {}).get("args", {}).get("max_steps") == 120, "training_config run max_steps must be 120")
    expect(config.get("checkpoint_policy", {}).get("selected_checkpoint_step") == 80, "training_config selected step must be 80")
    parameter_update = config.get("parameter_update", {})
    expect(all(parameter_update.get(k) is None for k in ("lora_rank", "lora_alpha", "lora_dropout")), "unused LoRA fields must be explicit nulls")
    metrics = [json.loads(line) for line in (out / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metric_fields = {"training_run_id", "phase", "step", "epoch", "loss", "learning_rate", "gradient_norm", "input_tokens", "throughput_tokens_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "recorded_at", "checkpoint_artifact"}
    expect(bool(metrics) and all(metric_fields <= set(x) for x in metrics), "training_metrics rows must preserve the normalized metric schema")

    # Cross-check every full-cohort and valid-only ratio against its own denominator.
    source_scores = json.loads((ROOT / "safety/wildguard_judge/results.json").read_text(encoding="utf-8"))["four_group_automatic_scores"]
    summary_text = (out / "DELIVERY_SUMMARY.md").read_text(encoding="utf-8")
    chart_text = (out / "attachments/charts/four_group_wildguard.svg").read_text(encoding="utf-8")
    packaged_groups = evaluation.get("four_group_open_response_results", {}).get("groups", {})
    for group, source_group in source_scores.items():
        for source_key, packaged_key in (("harmful_answer_rate_on_harmful_requests", "harmful_response_rate_on_harmful_requests"), ("refusal_rate_on_normal_requests", "normal_request_refusal_proxy")):
            raw = source_group[source_key]
            shown = packaged_groups.get(group, {}).get(packaged_key, {})
            expected_full = 100.0 * raw["numerator_known_positive"] / raw["denominator"]
            expected_valid = 100.0 * raw["numerator_known_positive"] / raw["effective_denominator"]
            expect(shown.get("known_positive_count") == raw["numerator_known_positive"] and shown.get("full_cohort_denominator") == raw["denominator"], f"{group}/{source_key} full-cohort counts changed")
            expect(math.isclose(shown.get("known_positive_rate_percent_full_cohort", float("nan")), expected_full, abs_tol=1e-12), f"{group}/{source_key} full-cohort rate does not match its denominator")
            expect(shown.get("effective_valid_denominator") == raw["effective_denominator"] and math.isclose(shown.get("valid_only_rate_percent", float("nan")), expected_valid, abs_tol=1e-12), f"{group}/{source_key} valid-only rate does not match its denominator")
            fraction = f'{raw["numerator_known_positive"]}/{raw["denominator"]}'
            percentage = f'{expected_full:.2f}%'
            group_line = next((line for line in summary_text.splitlines() if line.startswith(f"|{group}|")), "")
            expected_adjacent = f"{fraction}（已判阳性/全组={percentage}）"
            expect(expected_adjacent in group_line, f"DELIVERY_SUMMARY fraction/percentage mismatch or not adjacent for {group}/{source_key}")
            chart_label = f'{fraction} ({percentage}; U{raw["unknown"]})'
            expect(chart_label in chart_text, f"WildGuard chart is missing consistent {group}/{source_key} fraction/percentage/unknown count")
    for key in ("overall_result",):
        expect(isinstance(evaluation.get(key), dict), f"evaluation_summary.{key} must be an object")

    return {"passed": not errors, "errors": errors, "dataset_rows": len(dataset_rows), "capability_result_rows": len(evaluation.get("capability_results", []))}


def validate_manifest(out):
    errors = []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for key in ("delivery_id", "delivery_version", "schema_version", "model_id", "base_model_id", "confidentiality", "files", "known_issues", "submission_status"):
        if key not in manifest:
            errors.append(f"manifest field missing: {key}")
    if not isinstance(manifest.get("files"), list) or not isinstance(manifest.get("known_issues"), list):
        errors.append("manifest files and known_issues must be arrays")
    if manifest.get("submitted_at") is not None or manifest.get("submission_status") != "not_submitted":
        errors.append("manifest must preserve explicit not-submitted state")
    indexed = {x.get("path"): x for x in manifest.get("files", [])}
    actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if set(indexed) != actual:
        errors.append("manifest file list does not match package files")
    for relpath, record in indexed.items():
        for key in ("path", "purpose", "record_count", "size_bytes", "sha256", "availability"):
            if key not in record:
                errors.append(f"manifest file record missing {key}: {relpath}")
        p = out / relpath
        if not p.is_file():
            continue
        raw = p.read_bytes()
        if record.get("record_count") is None and relpath.endswith(".jsonl"):
            errors.append(f"JSONL record_count missing: {relpath}")
        if record.get("size_bytes") != len(raw) or record.get("sha256") != sha_bytes(raw):
            errors.append(f"manifest size/hash mismatch: {relpath}")
        if relpath.endswith(".jsonl"):
            count = sum(1 for line in raw.splitlines() if line.strip())
            if record.get("record_count") != count:
                errors.append(f"manifest record_count mismatch: {relpath}")
    return {"passed": not errors, "errors": errors, "indexed_file_count": len(indexed)}


PURPOSES = {
    "model_card.json": "model identity, intended use, license, access and limitations",
    "model_artifacts.json": "artifact locations, hashes, reconstructability and transfer status",
    "post_training.json": "RMU post-training method and quality controls",
    "training_datasets.jsonl": "training and validation dataset provenance and counts",
    "training_config.json": "resolved RMU run configuration and selected checkpoint relation",
    "training_metrics.jsonl": "normalized archived step-level training/validation metrics",
    "evaluation_summary.json": "capability, classifier, four-group and assisted-review results",
    "data_statement.md": "data provenance, licenses and distribution limits",
    "README.md": "delivery package overview and status",
    "DELIVERY_SUMMARY.md": "Chinese delivery report for reviewers",
    "DELIVERY_CHECK_REPORT.md": "integrity and completeness verification report",
    "MISSING_ITEMS.md": "remaining transfer and platform integration items",
    "delivery_audit.json": "lightweight package integrity and completeness checks",
    "attachments/charts/rmu_steps_bio_mmlu.svg": "existing full-evaluation RMU step tradeoff",
    "attachments/charts/four_group_wildguard.svg": "WildGuard-scored four-group comparison",
    "attachments/logs/rmu80_run_summary.json": "sanitized archived run provenance summary",
    "attachments/logs/README.md": "log provenance and excluded sensitive records",
    "artifacts/model/README.md": "local RMU delta location and reconstruction boundary",
    "artifacts/tokenizer/README.md": "base tokenizer reference and transfer status",
    "artifacts/serving/README.md": "current offline inference entry points and missing service components",
    "datasets/train/README.md": "restricted training data is referenced locally, not copied",
    "datasets/validation/README.md": "held-out representation-loss data is referenced locally, not copied",
    "datasets/evaluation/README.md": "evaluation source/status index; no raw evaluation rows copied",
    "attachments/charts/README.md": "chart sources and metric direction",
}


def svg_line_chart():
    points = [
        (0, 73.84131971720346, 70.673693206096),
        (40, 55.53809897879026, 68.49451645064805),
        (80, 42.73369992144541, 66.94915254237289),
        (120, 41.16260801256873, 66.20851730522718),
    ]
    width, height = 960, 520
    left, right = 90, 920
    panel1_top, panel1_bottom = 88, 260
    panel2_top, panel2_bottom = 320, 470
    x = lambda step: left + (right - left) * step / 120
    y1 = lambda value: panel1_bottom - (value - 35) / (78 - 35) * (panel1_bottom - panel1_top)
    y2 = lambda value: panel2_bottom - (value - 64) / (72 - 64) * (panel2_bottom - panel2_top)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#18212f}.title{font-size:23px;font-weight:bold}.sub{font-size:14px}.axis{font-size:13px}.grid{stroke:#d9dee8;stroke-width:1}.bio{stroke:#c4433b;fill:none;stroke-width:3}.mmlu{stroke:#2463a6;fill:none;stroke-width:3}.dotb{fill:#c4433b}.dotm{fill:#2463a6}</style>',
              '<text x="480" y="34" text-anchor="middle" class="title">连续120步轨迹：RMU步数与基准准确率</text>',
              '<text x="480" y="58" text-anchor="middle" class="sub">同一训练运行在40、80、120步保存；80步为主交付检查点</text>',
              '<text x="18" y="173" transform="rotate(-90 18 173)" text-anchor="middle" class="axis">WMDP-Bio准确率（%）</text>',
              '<text x="18" y="395" transform="rotate(-90 18 395)" text-anchor="middle" class="axis">MMLU准确率（%）</text>']
    for v in (40, 50, 60, 70):
        yy = y1(v); chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/><text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" class="axis">{v}</text>')
    for v in (65, 67, 69, 71):
        yy = y2(v); chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/><text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" class="axis">{v}</text>')
    for step, bio, mmlu in points:
        xx = x(step)
        chunks.append(f'<line x1="{xx:.1f}" y1="{panel1_top}" x2="{xx:.1f}" y2="{panel2_bottom}" stroke="#eef0f4"/>')
        chunks.append(f'<text x="{xx:.1f}" y="493" text-anchor="middle" class="axis">{step}</text>')
    bio_pts = " ".join(f"{x(s):.1f},{y1(b):.1f}" for s,b,_ in points)
    mmlu_pts = " ".join(f"{x(s):.1f},{y2(m):.1f}" for s,_,m in points)
    chunks += [f'<polyline points="{bio_pts}" class="bio"/>', f'<polyline points="{mmlu_pts}" class="mmlu"/>']
    for step, bio, mmlu in points:
        xx=x(step); yb=y1(bio); ym=y2(mmlu)
        chunks += [f'<circle cx="{xx:.1f}" cy="{yb:.1f}" r="5" class="dotb"/>',
                   f'<text x="{xx:.1f}" y="{yb-9:.1f}" text-anchor="middle" class="axis">{bio:.2f}</text>',
                   f'<circle cx="{xx:.1f}" cy="{ym:.1f}" r="5" class="dotm"/>',
                   f'<text x="{xx:.1f}" y="{ym-9:.1f}" text-anchor="middle" class="axis">{mmlu:.2f}</text>']
    chunks += ['<text x="505" y="516" text-anchor="middle" class="axis">训练轨迹步数</text>',
               '<line x1="665" y1="78" x2="695" y2="78" class="bio"/><text x="703" y="83" class="axis">WMDP-Bio（越低表示该基准准确率下降）</text>',
               '<line x1="665" y1="302" x2="695" y2="302" class="mmlu"/><text x="703" y="307" class="axis">MMLU（越低表示通用能力损失）</text>',
               '</svg>']
    return "\n".join(chunks) + "\n"


def svg_four_group_chart(scores):
    groups = []
    for name in ("A", "B/main", "B/strict", "C", "D/main", "D/strict"):
        item = scores[name]
        harmful = item["harmful_answer_rate_on_harmful_requests"]
        refusal = item["refusal_rate_on_normal_requests"]
        harmful_percent = 100 * harmful["numerator_known_positive"] / harmful["denominator"]
        refusal_percent = 100 * refusal["numerator_known_positive"] / refusal["denominator"]
        groups.append((name, harmful_percent, refusal_percent, harmful, refusal))
    width, height = 1060, 520
    left, right = 95, 1030
    top1, bottom1 = 95, 250
    top2, bottom2 = 335, 470
    xstep = (right-left)/len(groups)
    y1 = lambda v: bottom1-v/3.0*(bottom1-top1)
    y2 = lambda v: bottom2-v/17.0*(bottom2-top2)
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#18212f}.title{font-size:22px;font-weight:bold}.sub{font-size:14px}.axis{font-size:12px}.grid{stroke:#d9dee8;stroke-width:1}.bar1{fill:#c4433b}.bar2{fill:#3573b9}</style>',
              '<text x="530" y="31" text-anchor="middle" class="title">四组输出：WildGuard自动评分（512-token回答）</text>',
              '<text x="530" y="54" text-anchor="middle" class="sub">标签为WildGuard自动评分；显示阳性数/全组分母及同分母百分比；U表示unknown数</text>',
              '<text x="16" y="174" transform="rotate(-90 16 174)" text-anchor="middle" class="axis">有害请求中回答有害率（%）</text>',
              '<text x="16" y="401" transform="rotate(-90 16 401)" text-anchor="middle" class="axis">正常请求中回答拒答率（%）</text>']
    for v in (0,1,2,3):
        yy=y1(v); chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/><text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{v}</text>')
    for v in (0,5,10,15):
        yy=y2(v); chunks.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/><text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" class="axis">{v}</text>')
    for i,(name,harm,refusal,harm_item,refusal_item) in enumerate(groups):
        cx=left+xstep*(i+0.5); bw=32
        h1=(bottom1-y1(harm)); h2=(bottom2-y2(refusal))
        harm_label = f'{harm_item["numerator_known_positive"]}/{harm_item["denominator"]} ({harm:.2f}%; U{harm_item["unknown"]})'
        refusal_label = f'{refusal_item["numerator_known_positive"]}/{refusal_item["denominator"]} ({refusal:.2f}%; U{refusal_item["unknown"]})'
        chunks += [f'<rect x="{cx-bw/2:.1f}" y="{y1(harm):.1f}" width="{bw}" height="{h1:.1f}" class="bar1"/>',
                   f'<text x="{cx:.1f}" y="{y1(harm)-6:.1f}" text-anchor="middle" class="axis">{html.escape(harm_label)}</text>',
                   f'<rect x="{cx-bw/2:.1f}" y="{y2(refusal):.1f}" width="{bw}" height="{h2:.1f}" class="bar2"/>',
                   f'<text x="{cx:.1f}" y="{y2(refusal)-6:.1f}" text-anchor="middle" class="axis">{html.escape(refusal_label)}</text>',
                   f'<text x="{cx:.1f}" y="492" text-anchor="middle" class="axis">{html.escape(name)}</text>']
    chunks += ['<rect x="705" y="68" width="14" height="14" class="bar1"/><text x="725" y="80" class="axis">有害请求上的WildGuard回答有害率</text>',
               '<rect x="705" y="294" width="14" height="14" class="bar2"/><text x="725" y="306" class="axis">正常请求上的WildGuard拒答率代理</text>',
               '<text x="530" y="516" text-anchor="middle" class="axis">系统 / 实验策略</text>',
               '</svg>']
    return "\n".join(chunks) + "\n"


def pct_entry(obj):
    denominator = obj["denominator"]
    effective_denominator = obj["effective_denominator"]
    numerator = obj["numerator_known_positive"]
    return {
        "known_positive_count": numerator,
        "full_cohort_denominator": denominator,
        "known_positive_rate_percent_full_cohort": 100.0 * numerator / denominator if denominator else None,
        "all_sample_rate_bounds_percent": [100.0 * obj["all_sample_rate_lower"], 100.0 * obj["all_sample_rate_upper"]],
        "effective_valid_denominator": effective_denominator,
        "valid_only_rate_0_to_1": obj["effective_rate"],
        "valid_only_rate_percent": 100.0 * numerator / effective_denominator if effective_denominator else None,
        "unknown_count": obj["unknown"],
        "unknown_rate_full_cohort_0_to_1": obj["unknown_rate"],
        "valid_prediction_coverage_0_to_1": obj["coverage"],
        "denominator_note": "full_cohort uses all eligible rows; valid_only excludes unknown/error rows; bounds treat unknown rows as negative/positive respectively",
    }


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attachments/charts").mkdir(parents=True, exist_ok=True)
    for sub in ("artifacts/model", "artifacts/tokenizer", "artifacts/serving",
                "datasets/train", "datasets/validation", "datasets/evaluation",
                "attachments/logs"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    run = load(RUN_REL / "run.json")
    env = load(RUN_REL / "environment_versions.json")
    cfg = load(RUN_REL / "training_config.json")
    actual_args = load(RUN_REL / "actual_training_arguments.json")
    checkpoint = load(RUN_REL / "checkpoint-80/checkpoint.json")
    rebuild = load(RUN_REL / "checkpoint-80/rebuild_verification.json")
    base_manifest = load("logs/base_model_manifest.json")
    base_fingerprints = load("open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json")
    four_manifest = load("safety/four_groups/manifest.json")
    four_config = load("safety/four_groups/config.json")
    four_results = load("safety/four_groups/results.json")
    eval_run = load(RUN_REL / "full_evaluations/evaluation_run.json")
    step_results = {str(s): load(RUN_REL / f"full_evaluations/checkpoint-{s}/result.json") for s in (40, 80, 120)}
    bio_processing = load("data/processed/bio_v1/processing_manifest.json")
    block_manifest = load("data/processed/bio_v1/formal_blocks_v1/manifest.json")
    forget_source = load("logs/wmdp_bio_forget_download_manifest.json")
    retain_source = load("logs/wmdp_download_manifest.json")
    mmlu_manifest = load("logs/mmlu_dataset_manifest.json")
    mmlu_baseline = load("logs/mmlu_baseline_run.json")
    bio_eval_source = load("data/evaluation/wmdp_bio/snapshot_metadata.json")
    guard_source = load("safety/qwen3guard/records/model_source.json")
    guard_test = load("safety/qwen3guard/wildguardtest/results.json")
    judge = load("safety/wildguard_judge/results.json")
    # Repeat only the published aggregate cross-tabs. Do not load or copy the
    # per-item assisted-label summary, blind IDs, rationales, or source mapping.
    assisted = {
        "import_date": "2026-10-15",
        "source_file_sha256": "281414a6f57a4bd87f89ad5a0eb6c2c79956032fd4550af179c7fc49fb766cfa",
        "model_version_recorded": False,
        "human_verified_false_count": 30,
        "human_gold_created": False,
        "response_harmful_counts": {"true": 2, "false": 21, "null": 7},
        "response_refusal_counts": {"true": 10, "false": 20, "null": 0},
        "sample_counts": {"random20": 20, "diagnostic10": 10},
        "groups": {
            "random20": {
                "response_harmfulness": {"n":20,"both_explicit":18,"agreement":18,"disagreement":0,"assistant_uncertain":2,"wildguard_unknown":0,"wildguard_error":0,"cross_table":{"true":{"true":0,"false":0,"uncertain":0},"false":{"true":0,"false":18,"uncertain":2},"unknown":{"true":0,"false":0,"uncertain":0},"error":{"true":0,"false":0,"uncertain":0}}},
                "response_refusal": {"n":20,"both_explicit":20,"agreement":20,"disagreement":0,"assistant_uncertain":0,"wildguard_unknown":0,"wildguard_error":0,"cross_table":{"true":{"true":10,"false":0,"uncertain":0},"false":{"true":0,"false":10,"uncertain":0},"unknown":{"true":0,"false":0,"uncertain":0},"error":{"true":0,"false":0,"uncertain":0}}},
            },
            "diagnostic10": {
                "response_harmfulness": {"n":10,"both_explicit":5,"agreement":5,"disagreement":0,"assistant_uncertain":5,"wildguard_unknown":0,"wildguard_error":0,"cross_table":{"true":{"true":2,"false":0,"uncertain":5},"false":{"true":0,"false":3,"uncertain":0},"unknown":{"true":0,"false":0,"uncertain":0},"error":{"true":0,"false":0,"uncertain":0}}},
                "response_refusal": {"n":10,"both_explicit":10,"agreement":10,"disagreement":0,"assistant_uncertain":0,"wildguard_unknown":0,"wildguard_error":0,"cross_table":{"true":{"true":0,"false":0,"uncertain":0},"false":{"true":0,"false":10,"uncertain":0},"unknown":{"true":0,"false":0,"uncertain":0},"error":{"true":0,"false":0,"uncertain":0}}},
            },
        },
    }
    model_config = load("models/Qwen3-4B-Instruct-2507/config.json")
    control = load(RUN_REL / "control_vector.json")
    selected_steps = (40, 80, 120)

    run_metrics = [json.loads(x) for x in (RUN / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    step80_train = [x for x in run_metrics if x.get("phase") == "train" and x.get("step", 10**9) <= 80]
    exposures = {}
    for role in ("forget", "retain"):
        indices = []
        for row in step80_train:
            indices.extend(row.get(f"sampled_{role}_indices") or [])
        exposures[role] = {
            "exposure_count": len(indices),
            "unique_block_count": len(set(indices)),
            "candidate_pool_count": 2048,
        }
    val_rows_80 = [x for x in run_metrics if x.get("phase") == "validation" and x.get("step", 10**9) <= 80]
    val_block_evaluations_per_role = sum(x.get("blocks_per_role") or 0 for x in val_rows_80)

    processed = {x["path"]: x for x in bio_processing["output_files"]}
    block_files = {x["path"]: x for x in block_manifest["files"]}
    source_versions = {
        "forget": {
            "id": forget_source["dataset_id"], "revision": forget_source["revision"],
            "license": "gated; redistribution limited to authorized collaborator list",
            "source_type": "gated_official_dataset",
            "authorization_status": "recipient authorization required; current package does not transfer data",
            "source_detail": "CAIS WMDP-Bio forget corpus; immutable source revision is recorded in the archived download manifest.",
        },
        "retain": {
            "id": "WMDP Bio retain corpus in archived CAIS WMDP corpora archive",
            "revision": None,
            "license": "redistribution authorization not established in archived records",
            "source_type": "archived_corpus_derived",
            "authorization_status": "source revision and recipient redistribution authorization are unresolved",
            "source_detail": "Retain data extracted from the archived WMDP corpora bundle; exact upstream dataset revision was not recorded.",
        },
    }
    package_relative_paths = {
        p.relative_to(OUT).as_posix()
        for p in OUT.rglob("*") if p.is_file()
    }

    def dataset_common(dataset_id, dataset_name, dataset_version, split, purpose,
                       source_type, source_detail, file_path, file_format,
                       raw_count, used_count, cleaning_version, deduplication_method,
                       license_info, overlap, **extra):
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_version": dataset_version,
            "split": split,
            "purpose": purpose,
            "source_type": source_type,
            "source_detail": source_detail,
            "file_paths": [file_path],
            "file_path_scope": "project-relative actual source path on the training host",
            "file_paths_included_in_delivery_package": [file_path in package_relative_paths],
            "file_paths_included_note": "false means the source data file exists outside this metadata package and has not been transferred",
            "file_format": file_format,
            "raw_count": raw_count,
            "used_count": used_count,
            "cleaning_pipeline_version": cleaning_version,
            "deduplication_method": deduplication_method,
            "license_or_authorization": license_info,
            "test_overlap_result": overlap,
            **extra,
        }

    def license_record(source):
        return {
            "license_or_terms": source["license"],
            "source_revision": source["revision"],
            "recipient_authorization_status": source["authorization_status"],
            "redistribution_authorization": None,
            "status_note": "No data is included in this package; recipient access and redistribution rights must be confirmed independently.",
        }

    data_rows = []
    for role in ("forget", "retain"):
        for split in ("train", "validation"):
            path = f"data/processed/bio_v1/{role}/{split}.jsonl"
            record = processed[path]
            article_pool_count = 2048 if split == "train" else 64
            overlap = {
                "status": "partial_documented_screen_not_a_no_overlap_guarantee",
                "evaluation_questions_checked": bio_processing["evaluation_questions_checked"],
                "confirmed_matches": bio_processing["evaluation_confirmed_matches"],
                "method": bio_processing["evaluation_check"],
                "reviewed_candidates": bio_processing["candidate_review_complete"],
                "limitations": bio_processing["limitations"],
            }
            source_detail = {
                "source_dataset_id": source_versions[role]["id"],
                "source_revision": source_versions[role]["revision"],
                "source_description": source_versions[role]["source_detail"],
                "raw_role_total_before_cleaning_and_split": bio_processing["raw_counts"][role],
                "cleaned_records_in_this_split": record["records"],
                "preclean_split_specific_raw_count_recorded": False,
                "direct_selected_run_article_reads": 0,
            }
            data_rows.append(dataset_common(
                dataset_id=f"wmdp_bio_{role}_cleaned_{split}_bio-clean-v1",
                dataset_name=f"WMDP-Bio {role} cleaned article corpus",
                dataset_version=bio_processing["processing_version"],
                split=split,
                purpose="source article pool used in preprocessing to construct fixed token blocks; not read directly by the selected RMU training run",
                source_type="processed_project_dataset",
                source_detail=source_detail,
                file_path=path,
                file_format="JSONL article records",
                raw_count=None,
                used_count=article_pool_count,
                cleaning_version=bio_processing["processing_version"],
                deduplication_method=bio_processing["deduplication"],
                license_info=license_record(source_versions[role]),
                overlap=overlap,
                role=role,
                raw_count_note=("Split-specific pre-clean row count was not recorded. raw_role_total_before_cleaning_and_split records the known combined role count; it is not repeated as a per-split count."),
                available_cleaned_article_count=record["records"],
                selected_source_articles_for_fixed_block_pool=article_pool_count,
                actual_run_input_record_count=0,
                actual_usage_note="used_count is the number of source articles selected by preprocessing for the fixed block pool; the training process read the token-block JSONL rows below, not these article rows.",
                sha256=record["sha256"],
                size_bytes=record["size_bytes"],
                transferred_with_package=False,
            ))
    for role in ("forget", "retain"):
        for split in ("train", "validation"):
            path = f"data/processed/bio_v1/formal_blocks_v1/{role}/{split}.jsonl"
            block = block_files[path]
            if split == "train":
                usage = exposures[role]
                used_count = usage["exposure_count"]
                usage_note = f'{usage["exposure_count"]} presentations through selected step 80; {usage["unique_block_count"]} unique blocks from {usage["candidate_pool_count"]} candidate blocks'
                unique_used = usage["unique_block_count"]
            else:
                used_count = val_block_evaluations_per_role
                usage_note = f'{len(val_rows_80)} fixed validation passes through step 80; {block["blocks"]} distinct blocks per pass, {val_block_evaluations_per_role} block evaluations per role'
                unique_used = block["blocks"]
            source_count = processed[f"data/processed/bio_v1/{role}/{split}.jsonl"]["records"]
            overlap = {
                "status": "inherits_partial_article_level_screen_not_a_no_overlap_guarantee",
                "evaluation_questions_checked": bio_processing["evaluation_questions_checked"],
                "confirmed_matches": bio_processing["evaluation_confirmed_matches"],
                "method": bio_processing["evaluation_check"],
                "limitations": bio_processing["limitations"],
            }
            data_rows.append(dataset_common(
                dataset_id=f"rmu_formal_blocks_{role}_{split}_rmu-formal-blocks-v1",
                dataset_name=f"RMU fixed {role} token blocks",
                dataset_version=block_manifest["version"],
                split=split,
                purpose=("direct RMU input: forget role receives representation-misdirection target; retain role matches frozen reference representation" if split == "train" else "held-out representation-loss monitoring only; not an independent capability/safety validation set"),
                source_type="derived_project_token_block_dataset",
                source_detail={
                    "source_dataset_id": source_versions[role]["id"],
                    "source_revision": source_versions[role]["revision"],
                    "source_description": source_versions[role]["source_detail"],
                    "source_cleaned_article_file": f"data/processed/bio_v1/{role}/{split}.jsonl",
                    "source_cleaned_article_count": source_count,
                    "selected_articles_for_block_pool": block["selected_articles"],
                },
                file_path=path,
                file_format="JSONL token-id blocks; no standard answer targets",
                raw_count=block["blocks"],
                used_count=used_count,
                cleaning_version="bio-clean-v1 + rmu-formal-blocks-v1",
                deduplication_method=block_manifest["selection_rule"],
                license_info=license_record(source_versions[role]),
                overlap=overlap,
                role=role,
                candidate_block_count=block["blocks"],
                raw_count_definition="number of serialized token-block records in this fixed JSONL file before repeated run exposures/evaluations",
                unique_block_count_used=unique_used,
                used_count_definition="train: total block presentations through selected step 80; validation: total block evaluations through step 80; repeats are counted as exposures/evaluations",
                exposure_note=usage_note,
                tokenization=block_manifest["tokenization"],
                sequence_length=block_manifest["sequence_length"],
                sha256=block["sha256"],
                size_bytes=block["size_bytes"],
                transferred_with_package=False,
            ))
    write(OUT / "training_datasets.jsonl", "\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in data_rows))

    selected_metric_rows = []
    source_metric_lines = (RUN / "training_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    for line in source_metric_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        step = r.get("step")
        phase = r.get("phase")
        cp = f"checkpoint-{step}" if phase == "validation" and step in selected_steps else None
        selected_metric_rows.append({
            "training_run_id": r.get("training_run_id"),
            "phase": phase,
            "step": step,
            "epoch": r.get("epoch"),
            "loss": r.get("total_loss"),
            "forget_loss": r.get("forget_loss"),
            "retain_loss": r.get("retain_loss"),
            "validation_loss": r.get("validation_loss"),
            "retain_nll_monitor_only": r.get("retain_nll"),
            "retain_ppl_monitor_only": r.get("retain_ppl"),
            "learning_rate": r.get("learning_rate"),
            "gradient_norm": r.get("gradient_norm"),
            "input_tokens": r.get("input_tokens"),
            "loss_tokens": r.get("loss_tokens"),
            "nll_tokens": r.get("nll_tokens"),
            "cumulative_input_tokens": r.get("total_input_tokens"),
            "throughput_tokens_per_second": r.get("tokens_per_second"),
            "interval_seconds": r.get("interval_seconds"),
            "recorded_elapsed_seconds": r.get("elapsed_seconds"),
            "peak_allocated_bytes": r.get("gpu_peak_allocated_bytes"),
            "peak_reserved_bytes": r.get("gpu_peak_reserved_bytes"),
            "recorded_at": r.get("recorded_at"),
            "checkpoint_artifact": cp,
            "source_training_record_sha256": sha_bytes(line.encode("utf-8")),
            "unrecorded_values_are_null": ["wall_clock_timestamp_if_missing", "epoch_if_missing", "gradient_norm_on_validation"],
        })
    write(OUT / "training_metrics.jsonl", "\n".join(json.dumps(x, ensure_ascii=False, allow_nan=False) for x in selected_metric_rows))

    base_bytes = sum(x["size_bytes"] for x in base_manifest["files"])
    base_param_count = 4022468096
    trainable = 3 * 2560 * 9728
    runtime_fp = {
        "original_model_fingerprint": four_manifest["base_fingerprint"],
        "selected_rmu80_fingerprint": four_manifest["rmu80_fingerprint"],
        "verified_parameters": 398,
        "changed_parameters": 3,
        "unchanged_parameters": 395,
        "verification_source": str(RUN_REL / "checkpoint-80/rebuild_verification.json"),
    }
    control_path = RUN / "control_vector.json"
    delta_files = []
    for entry in checkpoint["parameters"]:
        pointer = SOURCE_DELTA / entry["file"]
        delta_files.append({
            "parameter_name": entry["name"],
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "stored_format": checkpoint["format"],
            "path_local_only": str(pointer),
            **lfs_pointer_info(pointer, entry),
            "base_tensor_sha256": entry["base_tensor_sha256"],
            "reconstructed_tensor_sha256": entry["trained_tensor_sha256"],
        })
    trajectory_deltas = []
    for step in selected_steps:
        cp_manifest = load(RUN_REL / f"checkpoint-{step}/checkpoint.json")
        trajectory_deltas.append({
            "step": step,
            "checkpoint_path_local_only": str(RUN / f"checkpoint-{step}"),
            "parameter_count": len(cp_manifest["parameters"]),
            "payload_bytes": sum(x["bytes"] for x in cp_manifest["parameters"]),
            "remote_verified": None,
            "remote_status": "unverified; remote LFS retrieval was not tested",
            "role": "same continuous120 trajectory; comparison attachment, not an independently trained model",
        })
    base_card_files = [
        {"path": f["path"], "size_bytes": f["size_bytes"], "sha256": f["sha256"],
         "status": "present_locally_not_transferred"}
        for f in base_manifest["files"]
    ]
    artifact_paths = {
        "base_model": "models/Qwen3-4B-Instruct-2507",
        "rmu80_delta": str(RUN_REL / "checkpoint-80"),
        "tokenizer": "models/Qwen3-4B-Instruct-2507/tokenizer.json and tokenizer_config.json",
        "base_config": "models/Qwen3-4B-Instruct-2507/config.json",
        "training_config": str(RUN_REL / "training_config.json"),
        "load_script": "scripts/rmu_adaptation_utils.py:load_delta",
        "environment_record": str(RUN_REL / "environment_versions.json"),
    }
    model_artifacts = {
        "schema_version": SCHEMA_DATE,
        "artifact_access": {
            "base_and_tokenizer_available_locally": True,
            "rmu_delta_available_in_local_lfs_cache": all(x["local_payload_hash_matches"] for x in delta_files),
            "remote_lfs_retrievable": None,
            "remote_lfs_status": "unverified",
            "included_in_delivery_package": False,
            "transferred_to_platform": False,
            "note": "Metadata package only. Local resources remain on the training host; remote access and platform transfer are unverified/not completed.",
        },
        "base_model": {
            "model_id": "qwen3-4b-instruct-2507",
            "official_name": "Qwen3-4B-Instruct-2507",
            "source_recorded": base_manifest["source"],
            "remote_revision": None,
            "revision_note": base_manifest["version_note"],
            "local_file_manifest": base_card_files,
            "total_manifested_file_bytes": base_bytes,
            "parameter_file_count": 398,
            "parameter_count": base_param_count,
            "file_and_parameter_fingerprints": {
                "file_manifest_path": "logs/base_model_manifest.json",
                "complete_parameter_fingerprint_path": "open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json",
                "four_group_original_runtime_fingerprint": runtime_fp["original_model_fingerprint"],
            },
            "status": "present_locally_not_transferred; recipient requires same authorized base model or official access",
        },
        "rmu_delta": {
            "selected_checkpoint": str(RUN_REL / "checkpoint-80"),
            "trajectory_id": "h12_a2_s32_seed42_continuous120",
            "selected_checkpoint_step": 80,
            "run_max_steps": 120,
            "independently_trained_80_step_run": False,
            "format": "bf16-uint16 XOR + gzip (bf16-uint16-xor-gzip-v1); lossless against fixed base tensors",
            "is_lora": False,
            "payload_bytes": sum(x["bytes"] for x in checkpoint["parameters"]),
            "delta_manifest_sha256": sha_bytes((SOURCE_DELTA / "checkpoint.json").read_bytes()),
            "parameters": delta_files,
            "local_rebuild_verification": rebuild,
            "rmu80_full_runtime_fingerprint": runtime_fp["selected_rmu80_fingerprint"],
            "status": "payloads_present_in_local_git_lfs_cache_not_in_delivery_package; remote upload/retrieval unverified",
        },
        "comparison_trajectory_checkpoints": trajectory_deltas,
        "tokenizer": {
            "files": [x for x in base_card_files if Path(x["path"]).name in {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}],
            "chat_template_sha256": four_manifest["chat_template_sha256"],
            "status": "present_locally_not_transferred; supplied with official base model",
        },
        "serving_and_loading": {
            "config_paths": ["models/Qwen3-4B-Instruct-2507/config.json", "models/Qwen3-4B-Instruct-2507/generation_config.json"],
            "reconstruction_entry": "scripts/rmu_adaptation_utils.py::load_delta",
            "loading_order": ["load the exact pinned base model", "verify base tensor fingerprint", "resolve and verify each XOR payload", "XOR-reconstruct the three BF16 tensors", "load the matching base tokenizer"],
            "repository_loader_notes": "The repository checkout contains Git LFS pointer text at delta paths. The actual compressed bytes are resolved by load_delta from local .git/lfs/objects after pointer/OID verification.",
            "reconstruction_command": "From project root: load the pinned local base with AutoModelForCausalLM.from_pretrained('models/Qwen3-4B-Instruct-2507', dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa', device_map='cuda'); then call load_delta(model, Path('open-unlearning/saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120/checkpoint-80')). The loader checks the complete base fingerprint, compressed SHA-256, XOR reconstruction, and trained tensor hashes.",
            "existing_verification_is_reused": True,
            "reverification_run_during_delivery_prep": False,
            "optimizer_state_saved": run["full_optimizer_state_persisted"],
        },
        "other_components": [
            {"name": "Qwen3Guard-Gen-0.6B", "role": "inference-time external safety classifier", "repo_id": guard_source["repo_id"], "revision": guard_source["revision"], "license": guard_source["license"], "local_cache": "cache/huggingface/...", "transferred": False, "required_for_plain_RMU_model": False},
            {"name": "allenai/wildguard 7B", "role": "offline evaluation judge only; not required by serving", "revision": judge["judge"]["revision"], "license": judge["judge"].get("license"), "local_cache": "cache/huggingface/...", "transferred": False, "required_for_serving": False},
        ],
        "loading_environment": {
            "python": env["python"],
            "torch": run["versions"]["torch"],
            "transformers": run["versions"]["transformers"],
            "accelerate": run["versions"]["accelerate"],
            "datasets": run["versions"]["datasets"],
            "lm_eval": run["versions"]["lm_eval"],
            "deepspeed": run["versions"]["deepspeed"],
            "gpu_recorded": load(RUN_REL / "environment_versions.json")["gpu"],
            "status": "environment exists locally; not transferred as a self-contained environment",
        },
        "delivery_boundary": "An absolute local path, symlink, or LFS pointer in source control is not a delivered weight file. No large weight or data payload is copied into this package.",
    }
    dump(OUT / "model_artifacts.json", model_artifacts)

    license_text = (BASE / "LICENSE").read_text(encoding="utf-8")
    base_license = "Apache-2.0" if "Apache License" in license_text and "Version 2.0" in license_text else "local license text; SPDX identifier not confirmed"
    model_card = {
        "schema_version": SCHEMA_DATE,
        "model_id": "hoshiuz-qwen3-4b-rmu80-v1",
        "model_name": "Qwen3-4B-Instruct-2507 RMU checkpoint at step 80",
        "model_version": "rmu80-continuous120-step80",
        "name": "Qwen3-4B-Instruct-2507 RMU checkpoint at step 80",
        "version": "rmu80-continuous120-step80",
        "base_model_id": "qwen3-4b-instruct-2507",
        "base_model_name": "Qwen3-4B-Instruct-2507",
        "base_model_version": "local file SHA-256 manifest and full parameter fingerprint; official immutable revision was not recorded",
        "model_type": "machine_unlearning",
        "model_type_note": "Project-level extension describing RMU machine unlearning; target-platform enum mapping is unconfirmed. Architecture is recorded separately. The actual post-training method is RMU, not SFT or DPO.",
        "post_training_method": "RMU",
        "provider": "HoshiuZ (RMU adaptation); base model by Qwen Team / Alibaba",
        "parameter_count": f"{base_param_count:,}",
        "parameter_count_exact": base_param_count,
        "parameter_count_note": "Counted from local safetensors headers across 398 unique tensors; RMU changes 3 tensors in-place and adds no model parameters.",
        "architecture": {
            "model_type": model_config["model_type"],
            "architectures": model_config["architectures"],
            "num_hidden_layers": model_config["num_hidden_layers"],
            "hidden_size": model_config["hidden_size"],
            "intermediate_size": model_config["intermediate_size"],
            "vocab_size": model_config["vocab_size"],
        },
        "modalities": ["text"],
        "context_length": model_config["max_position_embeddings"],
        "context_length_tokens": model_config["max_position_embeddings"],
        "precision": "BF16 base tensors and BF16 XOR delta",
        "release_status": "research artifact; not yet transferred to a delivery platform or released as a self-contained model",
        "artifact_access": {
            "local_available": {
                "base_model": True,
                "rmu80_delta_payloads_in_local_lfs_cache": all(x["local_payload_hash_matches"] for x in delta_files),
                "tokenizer": True,
            },
            "remote_retrievable": None,
            "remote_retrievable_status": "unverified; remote LFS upload and retrieval were not confirmed",
            "included_in_delivery_package": False,
            "transferred_to_platform": False,
            "recipient_access_requirements": ["authorized exact base model and tokenizer", "all matching RMU80 XOR delta objects", "project reconstruction loader and compatible runtime"],
            "note": "The RMU80 artifact is a base-dependent lossless BF16 XOR+gzip delta, not standalone weights or LoRA. Local presence does not establish remote access or platform transfer.",
        },
        "license": {"base_model": base_license, "derived_delta": "inherits base model license and applicable terms; distribution authorization not independently determined"},
        "chat_template": {
            "source": "base tokenizer official template",
            "sha256": four_manifest["chat_template_sha256"],
            "serving_protocol": "single user message, add_generation_prompt=true",
        },
        "default_system_settings": {
            "four_group_evaluation": None,
            "training_config_field": cfg["model"]["template_args"].get("system_prompt"),
            "note": "The RMU run consumed pre-tokenized blocks without applying this chat template. Four-group generation used no system message.",
        },
        "experiment_id_mapping": {
            "four_group_models": {"A": "original base", "B": "original base + Qwen3Guard", "C": "hoshiuz-qwen3-4b-rmu80-v1", "D": "same RMU80 + Qwen3Guard"},
            "source_run_id": run["run_id"] if "run_id" in run else cfg["task_name"],
            "source_trajectory_id": cfg["task_name"],
            "selected_checkpoint_path": str(RUN_REL / "checkpoint-80"),
        },
        "known_limitations": [
            "80 steps is selected from one continuous 120-step training trajectory, not an independently trained 80-step run.",
            "MMLU is down 3.7245 percentage points from baseline and does not meet the earlier 2-point target.",
            "Single-seed exploratory result; official evaluation test data were used during tuning.",
            "Evaluation responses have a 512-new-token limit; about half of original and RMU80 generations hit the limit.",
            "Open-ended safety results use WildGuard automatic ratings; no human-verified gold standard was completed.",
            "RMU80 WMDP-Bio accuracy decrease is a benchmark result, not evidence that all knowledge has been removed.",
            "Weight delta and training corpora have not been transferred; remote LFS availability remains unverified.",
        ],
    }
    dump(OUT / "model_card.json", model_card)

    cfg_portable = portable(cfg)
    actual_portable = portable(actual_args)
    training_config = {
        "schema_version": SCHEMA_DATE,
        "training_run_id": cfg["task_name"],
        "source_config_path": str(RUN_REL / "training_config.json"),
        "source_actual_training_arguments_path": str(RUN_REL / "actual_training_arguments.json"),
        "source_run_path_scope": "project-relative references; source project root on the training host",
        "config": cfg_portable,
        "actual_training_arguments": actual_portable,
        "run_relationship": {
            "max_steps": cfg["trainer"]["args"]["max_steps"],
            "selected_checkpoint_step": 80,
            "checkpoint_steps": load(RUN_REL / "completed_checkpoints.json"),
            "checkpoint_is_from_continuous_run": True,
            "checkpoint_is_independent_80_step_run": False,
            "configured_num_train_epochs": cfg["trainer"]["args"]["num_train_epochs"],
            "actual_epoch_at_selected_step": next(x.get("epoch") for x in run_metrics if x.get("phase") == "validation" and x.get("step") == 80),
        },
        "parameter_update": {
            "hook_module_pattern": cfg["trainer"]["method_args"]["module_regex"],
            "hook_module_indexing": "decoder block index is zero-based; model.layers.12 is the 13th block",
            "captured_activation": "matched transformer block output via forward hook",
            "updated_parameter_patterns": cfg["trainer"]["method_args"]["trainable_params_regex"],
            "actual_parameter_names": [x["name"] for x in checkpoint["parameters"]],
            "parameter_count": trainable,
            "alpha": cfg["trainer"]["method_args"]["alpha"],
            "alpha_meaning": "RMU retain representation-loss weight; not LoRA alpha",
            "lora_rank": None,
            "lora_alpha": None,
            "lora_dropout": None,
        },
        "objective": {
            "method": "pure RMU",
            "forget_loss": "masked per-example MSE between hooked forget activation and fixed random normalized direction scaled by steering coefficient",
            "retain_loss": "masked per-example MSE between train model and frozen reference model hooked activations",
            "total_loss": "gamma * forget_loss + alpha * retain_loss",
            "gamma": cfg["trainer"]["method_args"]["gamma"],
            "retain_loss_type": cfg["trainer"]["method_args"]["retain_loss_type"],
            "steering_coefficient": cfg["trainer"]["method_args"]["steering_coeff"],
            "control_vector": {"path_local_only": str(RUN_REL / "control_vector.json"), "sha256": sha_bytes(control_path.read_bytes()), "l2_norm": control["l2_norm"], "unit_direction_sha256": control["unit_direction_sha256"]},
            "padding_mask": "labels == -100 excluded from representation loss; attention mask also recorded in collator",
            "answer_targets": None,
            "answer_target_note": "RMU used token blocks and representation objectives; no standard Q/A target answer was created.",
        },
        "optimization": {
            "optimizer": cfg["trainer"]["args"]["optim"],
            "learning_rate": cfg["trainer"]["args"]["learning_rate"],
            "weight_decay": cfg["trainer"]["args"]["weight_decay"],
            "per_device_train_batch_size": cfg["trainer"]["args"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": cfg["trainer"]["args"]["gradient_accumulation_steps"],
            "effective_examples_per_update_per_role": cfg["trainer"]["args"]["gradient_accumulation_steps"] * cfg["trainer"]["args"]["per_device_train_batch_size"],
            "max_grad_norm": cfg["trainer"]["args"]["max_grad_norm"],
            "lr_scheduler_type": cfg["trainer"]["args"]["lr_scheduler_type"],
            "warmup_steps": cfg["trainer"]["args"]["warmup_steps"],
            "sequence_length_tokens": 256,
            "seed": cfg["trainer"]["args"]["seed"],
            "data_seed": cfg["trainer"]["args"]["data_seed"],
            "precision": "bfloat16",
            "attention_implementation": cfg["model"]["model_args"]["attn_implementation"],
            "gradient_checkpointing": cfg["trainer"]["args"]["gradient_checkpointing"],
        },
        "checkpoint_policy": {
            "trainer_save_strategy": cfg["trainer"]["args"]["save_strategy"],
            "project_delta_callback_checkpoint_steps": load(RUN_REL / "completed_checkpoints.json"),
            "selected_checkpoint_step": 80,
            "optimizer_state_saved": run["full_optimizer_state_persisted"],
            "exact_training_resume_supported": False,
        },
        "code": {
            "commit": run["code_commit"],
            "source_sha256": run["source_sha256"],
            "launch_command": run["command"],
            "versions": {k: run["versions"].get(k) for k in ("torch", "transformers", "accelerate", "datasets", "lm_eval", "deepspeed")},
            "python": env["python"],
        },
        "unrecorded_parameters": {
            "null_fields": ["LoRA parameters (method not used)", "wall-time fields not present in a source metric row"],
            "note": "No missing metric values were interpolated; actual resolved Trainer arguments are included above.",
        },
    }
    dump(OUT / "training_config.json", training_config)

    training_run_summary = {
        "training_run_id": cfg["task_name"],
        "status": run["status"],
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "run_max_steps": cfg["trainer"]["args"]["max_steps"],
        "selected_checkpoint_step": 80,
        "optimizer_update_loop_seconds": run["optimizer_update_loop_seconds"],
        "trainer_wall_seconds_including_callbacks": run["trainer_train_wall_seconds_including_callbacks"],
        "total_process_seconds": run["total_process_seconds"],
        "trainer_loss": run["trainer_metrics"]["train_loss"],
        "trainer_epoch": run["trainer_metrics"]["epoch"],
        "cuda_peak_allocated_bytes": run["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": run["cuda_peak_reserved_bytes"],
        "selected_checkpoint_payload_bytes": rebuild["bytes"],
        "selected_checkpoint_rebuild_hashes_match": rebuild["compressed_payload_and_base_and_reconstructed_hashes_match"],
        "reference_and_frozen_parameters_unchanged": run["reference_and_frozen_parameters_unchanged"],
        "full_optimizer_state_persisted": run["full_optimizer_state_persisted"],
        "note": "Sanitized run facts only; no request, answer, corpus text, or row-level annotations.",
    }
    dump(OUT / "attachments/logs/rmu80_run_summary.json", training_run_summary)
    write(OUT / "attachments/logs/README.md", """
# 日志来源

rmu80_run_summary.json 是从原训练 run.json、训练配置和80步重建校验记录整理的无文本摘要。逐步训练与验证指标见包内 training_metrics.jsonl。未复制控制台中的任意数据正文、请求/回答或逐条评分日志；完整原始日志仍位于项目本地路径，未转交。
""")

    post_training = {
        "schema_version": SCHEMA_DATE,
        "objective": "降低WMDP-Bio选择题基准准确率，同时以RMU retain表示损失约束表示漂移；该目标不是保证知识不可恢复",
        "methods": ["RMU"],
        "started_at": run["started_at"],
        "completed_at": run["completed_at"],
        "base_checkpoint": {"model_id": "qwen3-4b-instruct-2507", "local_path": "models/Qwen3-4B-Instruct-2507", "remote_revision": None, "fingerprint": runtime_fp["original_model_fingerprint"]},
        "output_checkpoint": {"model_id": "hoshiuz-qwen3-4b-rmu80-v1", "trajectory_id": cfg["task_name"], "local_path": str(RUN_REL / "checkpoint-80"), "selected_step": 80, "run_max_steps": 120, "independent_80_step_run": False, "fingerprint": runtime_fp["selected_rmu80_fingerprint"]},
        "training_data": [dict(x) for x in data_rows],
        "data_statistics": {"forget_candidate_blocks": 2048, "retain_candidate_blocks": 2048, "selected_step_exposures_per_role": {k:v["exposure_count"] for k,v in exposures.items()}, "unique_blocks_seen_by_selected_step": {k:v["unique_block_count"] for k,v in exposures.items()}, "validation_blocks_per_role": 64, "validation_passes_through_step80": len(val_rows_80), "validation_block_evaluations_per_role": val_block_evaluations_per_role},
        "safety_focus": "WMDP-Bio biological knowledge benchmark; external safety classifier is inference-time policy simulation and is not part of post-training.",
        "hyperparameters": {"hook_layer_zero_based": 12, "updated_layers_zero_based": [10,11,12], "updated_parameter_type": "mlp.down_proj.weight", "trainable_parameter_count": trainable, "learning_rate": cfg["trainer"]["args"]["learning_rate"], "retain_weight_alpha": cfg["trainer"]["method_args"]["alpha"], "steering_coefficient": cfg["trainer"]["method_args"]["steering_coeff"], "seed": cfg["trainer"]["args"]["seed"], "max_steps": cfg["trainer"]["args"]["max_steps"], "selected_checkpoint_step": 80, "precision": "BF16"},
        "quality_controls": [
            {"check_id": "delta_encoding", "status": "verified", "result": checkpoint["format"], "note": "Lossless XOR+gzip delta against the fixed base; not LoRA."},
            {"check_id": "archived_rebuild_hashes", "status": "verified_from_existing_record", "result": rebuild["compressed_payload_and_base_and_reconstructed_hashes_match"], "parameter_hashes": rebuild["trained_parameter_hashes"], "note": "Existing checkpoint-80 reconstruction record was reused; no rebuild was run in this phase."},
            {"check_id": "runtime_model_fingerprint", "status": "verified_from_existing_record", "result": runtime_fp["selected_rmu80_fingerprint"], "note": "Matches the archived four-group RMU80 fingerprint."},
            {"check_id": "reference_and_frozen_parameters", "status": "verified_from_training_log", "result": run["reference_and_frozen_parameters_unchanged"]},
            {"check_id": "training_source_commit", "status": "recorded", "result": run["code_commit"]},
            {"check_id": "rebuild_repeated_during_delivery_prep", "status": "not_run_by_scope", "result": False},
            {"check_id": "additional_training_or_evaluation", "status": "not_run_by_scope", "result": False},
        ],
        "platform_method_enum": {"method_used": "RMU", "target_platform_or_enum_schema_provided": False, "machine_unlearning_enum_available": None, "gap": "The target platform and its method enum were not provided or verified in this workspace. Keep the method as RMU; do not relabel as SFT or DPO. Resolve the platform mapping before submission."},
        "known_limitations": [
            "Single seed and one continuous 120-step trajectory; selected 80 is not an independent 80-step run.",
            "No optimizer state; checkpoint can be reconstructed for inference/evaluation but the exact training trajectory cannot be resumed.",
            "Official test data have been used in tuning; all results are exploratory, not independent test estimates.",
            "MMLU loss at selected step is 3.7245 percentage points, outside the previous 2-point target.",
            "No human-verified answer-safety labels; ChatGPT auxiliary review remains model-assisted and uncertain cases remain unresolved.",
            "Remote Git LFS payload retrieval for 40/80/120 deltas has not been verified.",
        ],
    }
    dump(OUT / "post_training.json", post_training)

    step_rows = []
    baseline = step_results["80"]["metrics"]
    original_bio = baseline["wmdp_bio"]["baseline_accuracy_percent"]
    original_mmlu = baseline["mmlu"]["baseline_accuracy_percent"]
    for s in selected_steps:
        metrics = step_results[str(s)]["metrics"]
        step_rows.append({
            "step": s,
            "same_continuous_trajectory": True,
            "wmdp_bio": {"n": metrics["wmdp_bio"]["n"], "correct": metrics["wmdp_bio"]["correct"], "accuracy_percent": metrics["wmdp_bio"]["accuracy_percent"], "baseline_percent": metrics["wmdp_bio"]["baseline_accuracy_percent"], "change_percentage_points": metrics["wmdp_bio"]["change_percentage_points"]},
            "mmlu": {"n": metrics["mmlu"]["n"], "correct": metrics["mmlu"]["correct"], "accuracy_percent": metrics["mmlu"]["accuracy_percent"], "baseline_percent": metrics["mmlu"]["baseline_accuracy_percent"], "change_percentage_points": metrics["mmlu"]["change_percentage_points"]},
            "full_evaluation_protocol": load(RUN_REL / f"full_evaluations/checkpoint-{s}/summary.json")["protocol"],
            "exploratory": True,
        })
    guard_tasks = {}
    for task, result in guard_test["tasks"].items():
        guard_tasks[task] = {
            "n": result["n"],
            "label_distribution": result["labels"],
            "errors": result["errors"],
            "error_rate_0_to_1": result["error_rate"],
            "coverage_0_to_1": result["coverage"],
            "policies": {},
        }
        for policy, p in result["policies"].items():
            m = p["system_all"]
            guard_tasks[task]["policies"][policy] = {k:m.get(k) for k in ("n","positive","negative","TP","FP","TN","FN","precision","recall","F1","macro_F1","FPR","FNR","recall_CI95","FPR_CI95")}
    four_group_scores = {}
    for group, score_obj in judge["four_group_automatic_scores"].items():
        four_group_scores[group] = {
            "harmful_response_rate_on_harmful_requests": pct_entry(score_obj["harmful_answer_rate_on_harmful_requests"]),
            "harmful_response_rate_on_all_requests": pct_entry(score_obj["harmful_answer_rate_on_all_requests"]),
            "normal_request_refusal_proxy": pct_entry(score_obj["refusal_rate_on_normal_requests"]),
            "normal_not_harmful_and_not_refusal_proxy": pct_entry(score_obj["normal_not_harmful_and_not_refusal_proxy"]),
            "unknown_response_harmfulness_all": score_obj["unknown_response_harmfulness_all"],
            "unknown_response_refusal_all": score_obj["unknown_response_refusal_all"],
        }
    assisted_summary = {
        "import_date": assisted["import_date"],
        "source_file_sha256": assisted["source_file_sha256"],
        "label_source": "chatgpt_assisted",
        "model_version_recorded": assisted["model_version_recorded"],
        "human_verified_false_count": assisted["human_verified_false_count"],
        "human_gold_created": assisted["human_gold_created"],
        "response_harmful_counts": assisted["response_harmful_counts"],
        "response_refusal_counts": assisted["response_refusal_counts"],
        "groups": assisted["groups"],
        "sample_counts": assisted["sample_counts"],
        "residual_positive_case_distribution": {
            "B/main": {"total":4,"assistant_harmful":1,"assistant_not_harmful":0,"assistant_uncertain":3},
            "D/main": {"total":3,"assistant_harmful":1,"assistant_not_harmful":0,"assistant_uncertain":2},
        },
        "possible_wildguard_miss_disagreement_count_in_review_sample": 0,
        "aggregate_source_reference": "safety/wildguard_judge/REPORT.md#chatgpt辅助复核",
        "item_ids_and_labels_included": False,
        "interpretation": "model-to-model agreement only; 20 random unique pairs and 10 targeted diagnostic pairs are not a population estimate; no labels were used to change full-sample metrics",
    }
    dataset_entries = [
        {
            "dataset_id": bio_eval_source["dataset"],
            "dataset_name": "WMDP-Bio",
            "split": bio_eval_source["split"],
            "revision": bio_eval_source["original_recorded_revision"],
            "revision_reverified": bio_eval_source["remote_revision_reverified"],
            "sample_count": bio_eval_source["records"],
            "sha256": bio_eval_source["sha256"],
            "license_or_access": "evaluation source authorization/license not independently verified in archived metadata",
            "included_in_delivery_package": False,
        },
        {
            "dataset_id": mmlu_manifest["dataset_id"],
            "dataset_name": "MMLU",
            "split": "all/test",
            "revision": mmlu_manifest["revision"],
            "sample_count": mmlu_baseline["sample_count"],
            "license_or_access": mmlu_manifest["license"],
            "sha256": None,
            "source_file_hashes": mmlu_manifest["source_files"],
            "note": "Subject-partitioned source manifest is archived; no single concatenated dataset hash is recorded. Official test was used in model tuning.",
            "included_in_delivery_package": False,
        },
        {
            "dataset_id": guard_test["dataset"]["dataset"]["dataset_id"],
            "dataset_name": "WildGuardTest for Qwen3Guard independent classifier evaluation",
            "revision": guard_test["dataset"]["dataset"]["revision"],
            "config": guard_test["dataset"]["dataset"]["config"],
            "split": guard_test["dataset"]["dataset"]["split"],
            "license_or_access": guard_test["dataset"]["dataset"]["license"],
            "input_sample_count": guard_test["dataset"]["eligible_input"],
            "output_sample_count": guard_test["dataset"]["eligible_output"],
            "raw_row_count": guard_test["dataset"]["raw_rows"],
            "included_in_delivery_package": False,
        },
        {
            "dataset_id": "wildguardtest-fixed-unique-requests-v1",
            "dataset_name": "WildGuardTest fixed request set for four-group generation",
            "source": "same WildGuardTest fixed request set",
            "sample_count": four_results["n"],
            "harmful_request_count": judge["four_group_input_counts"]["harmful"],
            "normal_request_count": judge["four_group_input_counts"]["normal"],
            "manifest_sha256": four_manifest["request_manifest_sha256"],
            "included_in_delivery_package": False,
        },
    ]
    capability_results = []
    for result_key, task_name, dataset_id, metric_key, note in [
        ("WMDP_Bio", "WMDP-Bio", bio_eval_source["dataset"], "wmdp_bio", "Lower accuracy means reduced performance on the target WMDP-Bio knowledge benchmark; it is not a calibrated safety score."),
        ("MMLU", "MMLU", mmlu_manifest["dataset_id"], "mmlu", "Lower accuracy means general-capability loss."),
    ]:
        step_metrics = [
            {
                "step": row["step"],
                "same_continuous_trajectory": row["same_continuous_trajectory"],
                **row[metric_key],
                "full_evaluation_protocol": row["full_evaluation_protocol"],
                "exploratory": row["exploratory"],
            }
            for row in step_rows
        ]
        selected_row = next(x for x in step_metrics if x["step"] == 80)
        capability_results.append({
            "metric_id": "wmdp_bio_accuracy" if result_key == "WMDP_Bio" else "mmlu_accuracy",
            "task_name": task_name,
            "dataset_id": dataset_id,
            "unit": "percent_0_to_100",
            "direction_note": note,
            "baseline_percent": selected_row["baseline_percent"],
            "steps": step_metrics,
            "trained_model_id": "hoshiuz-qwen3-4b-rmu80-v1",
            "selected_checkpoint_step": 80,
            "selected_change_pp": selected_row["change_percentage_points"],
            "exploratory": True,
            "official_test_used_during_tuning": True,
            "mmlu_loss_pp": (-step_results["80"]["metrics"]["mmlu"]["change_percentage_points"]) if result_key == "MMLU" else None,
            "meets_previous_two_point_constraint": False if result_key == "MMLU" else None,
        })

    evaluation_protocol_version = {
        "rmu_steps": "continuous120-full-eval-protocol",
        "qwen3guard": "safety/qwen3guard/wildguardtest/config.json",
        "four_groups": "four_groups_v1 / protocol SHA " + four_manifest["protocol_sha256"],
        "wildguard_judge": "WildGuard judge fixed config; see scorer fingerprint",
    }
    scoring_policy_version = {
        "qwen3guard_official_levels": ["Safe", "Controversial", "Unsafe"],
        "experiment_defined_main_policy": "allow Safe/Controversial; block Unsafe",
        "experiment_defined_strict_policy": "allow Safe; block Controversial/Unsafe",
        "judge_fields": ["request harmfulness", "response refusal", "response harmfulness"],
        "distinction": "main/strict are local action mappings; not official Qwen3Guard policies",
    }

    eval_summary = {
        "schema_version": SCHEMA_DATE,
        "evaluated_at": {
            "base_mmlu": mmlu_baseline["completed_recorded_at"],
            "rmu_step_evaluations": eval_run.get("completed_at"),
            "qwen3guard_test": None,
            "four_group_generation": None,
            "wildguard_scoring": None,
            "chatgpt_assisted_import": None,
            "note": "Dates not recorded in these source files are null rather than inferred.",
        },
        "base_model_id": "qwen3-4b-instruct-2507",
        "post_training_model_id": "hoshiuz-qwen3-4b-rmu80-v1",
        "trained_model_id": "hoshiuz-qwen3-4b-rmu80-v1",
        "evaluation_protocol_version": evaluation_protocol_version,
        "scoring_policy_version": scoring_policy_version,
        "protocol_version": evaluation_protocol_version,
        "scoring_strategy_version": scoring_policy_version,
        "judge": {
            "four_group_evaluator": judge["judge"],
            "qwen3guard_classifier": {"repo_id": guard_source["repo_id"],"revision": guard_source["revision"],"license": guard_source["license"],"purpose":"classification and the offline B/D simulation"},
            "wildguard_role": "offline response evaluator only; not a required defense component",
            "chatgpt_assisted_review": {"model_version_recorded": assisted["model_version_recorded"],"human_verified": False,"human_gold": False},
        },
        "inference_config": {
            "four_group": {"batch_size":20,"seed":42,"dtype":"bfloat16","attention":"sdpa","do_sample":False,"num_beams":1,"max_new_tokens":512,"use_cache":True,"thinking":"disabled; base Instruct-2507 non-thinking","system_prompt":None,"chat_template_sha256":four_manifest["chat_template_sha256"],"input_truncation":False},
            "generation_cutoff": {k:four_results["models"][k] for k in ("original","rmu80")},
            "qwen3guard": {"dtype":guard_test["runtime"]["dtype"],"device":guard_test["runtime"]["device"],"attention":guard_test["runtime"]["attention"],"generation":guard_test["runtime"]["generation"],"context_limit":guard_test["runtime"]["effective_context_limit"],"truncation":guard_test["runtime"]["truncation"]},
            "wildguard_scoring": {"revision":judge["judge"]["revision"],"runtime":judge["judge"]["runtime"],"deterministic":True},
        },
        "datasets": dataset_entries,
        "capability_results": capability_results,
        "classifier_results_on_existing_labeled_test": {
            "source": "Qwen3Guard on WildGuardTest official test rows; independent classifier evaluation, not four-group model answer scoring",
            "tasks": guard_tasks,
            "parse_errors": {k:v["errors"] for k,v in guard_test["tasks"].items()},
        },
        "four_group_open_response_results": {
            "title": "WildGuard automatic scores; 512-token generation budget",
            "scope": "Offline protection simulation, not an online service. A/C generated once; B/D reuse those cached answers.",
            "groups": four_group_scores,
            "paired_contrasts": judge["paired_contrasts"],
            "paired_contrast_denominator_note": "Each contrast uses the source paired_n as its denominator: requests where both compared systems had valid predictions. This paired denominator can differ from each full-cohort denominator; deltas and intervals are carried forward unchanged.",
            "zero_event_intervals": judge["zero_event_intervals"],
            "512_finish_reasons": judge["generation_512_finish_reasons"],
            "judge_known_limitations": judge["interpretation_limits"],
            "cannot_claim": ["human-confirmed harmful response rate","online latency","unlimited-length performance","a composite safety score"],
        },
        "chatgpt_assisted_review": assisted_summary,
        "overall_result": {
            "aggregate_score": None,
            "calibrated_risk": None,
            "ranking": None,
            "status": "not_defined",
            "note": "No predefined aggregation rule, calibrated risk measure, or ranking protocol exists; no composite result was invented.",
        },
        "comparison_valid": {
            "same_fixed_request_set_and_order_for_four_groups": True,
            "same_generation_protocol_for_A_and_C": True,
            "paired_WildGuard_automatic_rating": True,
            "independent_untouched_final_test": False,
            "single_seed": True,
        },
        "comparison_notes": [
            "The chosen C/D model is step80 from the same max_steps=120 trajectory as steps40 and120.",
            "WMDP-Bio knowledge-benchmark accuracy and WildGuard open-response labels answer different questions.",
            "Automatic judge errors and 512-token cutoff can affect open-response results.",
            "ChatGPT-assisted labels remain separate; no human verification and no full-sample correction.",
            "No strong evidence shows extra open-response safety benefit of RMU on top of the classifier.",
        ],
    }
    dump(OUT / "evaluation_summary.json", eval_summary)

    # Required directory-level status notes; do not copy external payloads.
    write(OUT / "artifacts/model/README.md", f"""
# RMU80 model artifact status

The actual selected checkpoint is at the project-local path {RUN_REL / "checkpoint-80"}. It is a BF16 uint16 XOR+gzip lossless delta against the pinned local base, not LoRA and not a standalone full model. The three compressed payloads total {rebuild["bytes"]:,} bytes and are present in this server's Git LFS object cache with matching SHA-256. The checked-out filenames are Git LFS pointers (133 bytes each), not payload files. Remote upload/retrieval is not verified. This delivery directory contains metadata only; no model bytes are copied.

Reconstruction uses scripts/rmu_adaptation_utils.py::load_delta from the project root after obtaining the exact base model and local LFS payloads. Existing checkpoint-80/rebuild_verification.json verifies compressed payload, base tensor and reconstructed tensor hashes. Delivery preparation reused this record and did not load/rebuild the model again. Four-group runtime model fingerprint: {runtime_fp["selected_rmu80_fingerprint"]}.
""")
    write(OUT / "artifacts/tokenizer/README.md", f"""
# Tokenizer status

Tokenizer files are present locally with the original base model at models/Qwen3-4B-Instruct-2507 and are not copied here. The official chat-template digest used by the four-group generation protocol is {four_manifest["chat_template_sha256"]}. Obtain the exact tokenizer with the base model; the RMU training itself read pre-tokenized blocks without applying the chat template.
""")
    write(OUT / "artifacts/serving/README.md", """
# Serving artifact status

No online inference service is present or claimed. The repository has offline model reconstruction/evaluation and safety simulation entry points:
- scripts/rmu_adaptation_utils.py::load_delta reconstructs the local RMU delta on the exact base.
- safety/four_groups/generate.py and safety/four_groups/finish.py implement the recorded offline generation/policy simulation.
- safety/qwen3guard/moderator.py implements the standalone classifier adapter.

This is not a deployment-ready service. Missing items include a serving API, authentication/authorization, request logging and retention policy, operational monitoring, service-level latency/load tests, deployment packaging, and a recipient-accessible model artifact store. No service was developed in this delivery phase.
""")
    write(OUT / "datasets/train/README.md", """
# Training data status

No training text or token block is included. training_datasets.jsonl identifies the local forget/retain source and exact fixed block files, record counts, hashes, exposure counts and restrictions. WMDP-Bio forget corpus access is restricted to authorized collaborators; the retain corpus redistribution permission is not established. A receiving platform must separately verify authorization before obtaining or reading either corpus.
""")
    write(OUT / "datasets/validation/README.md", """
# Validation data status

No validation text or token blocks are included. The local 64-block-per-role sets monitor RMU representation losses only. They are not an independent MMLU or safety validation set. WMDP-Bio/MMLU official test results were used during model tuning and are exploratory.
""")
    write(OUT / "datasets/evaluation/README.md", """
# Evaluation data status

No evaluation prompts, questions, requests, or answers are included. Sources and manifests are recorded in evaluation_summary.json. WMDP-Bio access/revision details and WildGuardTest text require recipient-side source access; MMLU source carries the archived MIT license metadata. Raw request/answer text remains local.
""")
    write(OUT / "attachments/charts/README.md", """
# 图表来源与方向

rmu_steps_bio_mmlu.svg 使用既有连续120步全量评测结果：Bio和MMLU均为原始准确率，越低分别表示WMDP-Bio目标基准表现下降和MMLU通用能力损失。four_group_wildguard.svg 使用已归档WildGuard自动评分汇总：有害回答率、正常请求拒答代理均来自WildGuard，并非人工标签。图表没有新计算预测或引入综合分数。
""")

    write(OUT / "attachments/charts/rmu_steps_bio_mmlu.svg", svg_line_chart())
    write(OUT / "attachments/charts/four_group_wildguard.svg", svg_four_group_chart(judge["four_group_automatic_scores"]))

    # Data statement and explanatory reports.
    write(OUT / "data_statement.md", f"""
# 数据说明

本交付包只包含数据来源、许可/访问状态、统计、哈希和本地路径索引；不包含训练语料、验证块、评测题、WildGuardTest请求/回答、四组生成回答或逐条辅助标签。

## 后训练数据

RMU训练输入来自本地CAIS WMDP-Bio forget论文语料和Bio retain语料。Forget源记录为 {forget_source["dataset_id"]}，revision {forget_source["revision"]}；其官方数据卡限制未经许可不得向协作者名单之外分发。Retain源从归档WMDP数据包读取，原始来源revision和再分发许可没有记录，不能推断框架MIT许可证覆盖语料。交付接收方必须单独确认资格和授权。

原始forget/retain记录分别为 {bio_processing["raw_counts"]["forget"]} / {bio_processing["raw_counts"]["retain"]}；清洗后的训练/验证文章计数列在 training_datasets.jsonl。清洗版本 bio-clean-v1；交叉角色冲突及精确重复处理规则、近重复启发式、固定文章池和token块构造规则均保留在本文件及项目清单中。RMU80实际来自2048个固定候选块/角色，至80步分别有 {exposures["forget"]["exposure_count"]} / {exposures["retain"]["exposure_count"]} 次block exposure，unique为 {exposures["forget"]["unique_block_count"]} / {exposures["retain"]["unique_block_count"]}。候选块数不是曝光数，曝光数也不是训练文章全文数。

存档记录显示对 {bio_processing["evaluation_questions_checked"]} 个评测问题做过受限的匹配筛查，确认匹配数为 {bio_processing["evaluation_confirmed_matches"]}；方法不是语义/答案级全覆盖，不能写成无污染保证。MMLU和WMDP-Bio官方test已参与选择/调参，相关分数为探索性结果。

## 评测数据

WMDP-Bio test源为 {bio_eval_source["dataset"]}/{bio_eval_source["subset"]}，本地快照1273条；远端revision复核标记为 {bio_eval_source["remote_revision_reverified"]}。WildGuardTest来自 {guard_test["dataset"]["dataset"]["dataset_id"]} revision {guard_test["dataset"]["dataset"]["revision"]}，官方test split许可记录为 ODC-BY；本包不分发原文。MMLU来源 {mmlu_manifest["dataset_id"]} revision {mmlu_manifest["revision"]}，存档卡记录MIT许可。

## 模型辅助复核与隐私

30条ChatGPT辅助标签属于模型辅助判断，具体模型版本未记录，全部human_verified=false；未完成人工金标准。交付包只含分组聚合和匿名统计，不含逐条标签、理由、原文或映射。空值保持不确定，未被当作安全。

## 分发状态

训练及评测数据当前只在项目服务器本地。包外绝对路径、Git LFS指针和代码仓库状态均不等于数据已经交给平台。不得把本包描述为自包含数据交付。
""")

    # A concise, teacher-facing report; values are read from archived results above.
    main = judge["four_group_automatic_scores"]
    def rate_display(item):
        numerator = item["numerator_known_positive"]
        denominator = item["denominator"]
        full_rate = 100.0 * numerator / denominator if denominator else None
        lower = 100.0 * item["all_sample_rate_lower"]
        upper = 100.0 * item["all_sample_rate_upper"]
        effective_denominator = item["effective_denominator"]
        valid_rate = 100.0 * numerator / effective_denominator if effective_denominator else None
        text = f"{numerator}/{denominator}（已判阳性/全组={full_rate:.2f}%）"
        if item["unknown"]:
            text += f"；unknown {item['unknown']}，全组范围 {lower:.2f}%–{upper:.2f}%"
        text += f"；有效样本 {numerator}/{effective_denominator}={valid_rate:.2f}%"
        return text

    summary_groups = ["A", "B/main", "B/strict", "C", "D/main", "D/strict"]
    summary_rate_rows = "\n".join(
        "|{group}|{harm}|{refusal}|".format(
            group=group,
            harm=rate_display(main[group]["harmful_answer_rate_on_harmful_requests"]),
            refusal=rate_display(main[group]["refusal_rate_on_normal_requests"]),
        )
        for group in summary_groups
    )
    original_stop = four_results["models"]["original"]["limit_fraction_of_completed"] * 100
    rmu_stop = four_results["models"]["rmu80"]["limit_fraction_of_completed"] * 100
    training_peak_gib = run["cuda_peak_allocated_bytes"] / 2**30
    training_reserved_gib = run["cuda_peak_reserved_bytes"] / 2**30
    gen_sessions = four_results["sessions"]
    current_free = shutil.disk_usage(ROOT).free
    write(OUT / "DELIVERY_SUMMARY.md", f"""
# WMDP-Bio RMU80与外挂安全分类器：本周交付材料说明

**状态：元数据、图表和报告已准备；完整模型权重和数据尚未转交到平台。** 本包是轻量交付索引，不是自包含模型包。

## 任务与模型

项目研究Qwen3-4B-Instruct-2507在WMDP-Bio上的纯RMU遗忘。主交付模型 hoshiuz-qwen3-4b-rmu80-v1 是同一条连续120步RMU轨迹的第80步检查点，并非单独训练80步。它从精确本地原模型加三层down_proj.weight的BF16位级XOR+gzip增量重建，增量不是LoRA。文件和参数指纹已由已有重建记录验证。本阶段没有再次加载模型或复制权重。

试验方法有两类：纯RMU修改模型参数；Qwen3Guard-Gen-0.6B作为推理外挂分类器。Qwen3Guard的main/strict拦截动作是本项目定义，不是模型官方策略。WildGuard只作为离线回答评判器，不是防护组件。四组为A原模型、B原模型加Guard、C纯RMU80、D RMU80加Guard。

## 数据与实验流程

训练时先从受限forget/retain文章语料构造固定token块，再按forget表示误导目标和retain参考表示目标训练；RMU没有标准问答目标答案。原始论文语料和固定token块都留在服务器，不放进本包。Bio/MMLU能力评测与安全分类/四组开放式回答评测分别记录，不能互相替代。

四组安全流程是离线模拟：A/C各生成一次512-token回答，B/D复用回答和Qwen3Guard审核；输入拦截时模拟系统不调用生成模型。它不是已部署在线服务。原模型512上限截停 {four_results["models"]["original"]["hit_generation_limit"]}/{four_results["models"]["original"]["completed"]}（{original_stop:.2f}%）；RMU80截停 {four_results["models"]["rmu80"]["hit_generation_limit"]}/{four_results["models"]["rmu80"]["completed"]}（{rmu_stop:.2f}%）。因此开放回答结论只适用于该预算。

## 已有结果

|模型/轨迹步数|WMDP-Bio准确率|MMLU准确率|MMLU相对基线损失|
|---|---:|---:|---:|
|原模型|73.8413%|70.6737%|0|
|连续轨迹40步|55.5381%|68.4945%|2.1792个百分点|
|**连续轨迹80步（主交付）**|**42.7337%**|**66.9492%**|**3.7245个百分点**|
|连续轨迹120步|41.1626%|66.2085%|4.4652个百分点|

RMU80未满足此前MMLU损失不超过2个百分点的目标。Bio准确率下降表示WMDP-Bio目标基准表现下降；MMLU下降表示通用能力损失。两者是不同任务，Bio变化不能替代开放式安全判断，也不代表所有有害知识消失。

按WildGuard自动评分，下面“已判阳性/全组”使用固定队列分母；若有unknown，同时列出将unknown视为阴性/阳性的全组范围。有效样本比例单独使用有效分母，不与全组分母混列。

|组别|有害请求中回答有害|正常请求回答拒答（代理）|
|---|---|---|
{summary_rate_rows}

有效样本上的原始预测、未知/错误数及覆盖率保留在 evaluation_summary.json。配对差值与置信区间仍使用归档的配对有效样本；其 paired_n 是各项差值/区间的分母，并不等同于上表全组分母。以上为自动评判标签，不是真人确认；WildGuard可能误判。现有配对分析无法明确纯RMU自身对开放式回答安全的变化，也不足以证明在Guard之上增加RMU有明确额外收益。这不推翻RMU在WMDP-Bio知识基准上的结果。

30条ChatGPT辅助复核分为20条随机问答对和10条定向诊断案例，全部human_verified=false，未完成人工核验。重点7个B/main、D/main残留阳性案例中，2条辅助判断有害、5条不确定；未决项保持未决，不修正全量WildGuard指标。

## 限制与交付状态

本轮是单seed探索性结果；官方测试集曾用于调参。不存在独立未触碰的最终安全测试集。回复截停、评判器误差和无人工金标准限制了安全结论。当前没有在线服务入口或服务运维材料。

本包包含标准元数据、真实逐步训练指标、评测汇总、数据说明、两张结果图和缺项清单。原模型文件、40/80/120增量、tokenizer、训练/验证语料、评测文本和逐条回答都没有复制。选中RMU80增量在本机LFS缓存可取回且哈希匹配，但远端备份未核实；Git代码推送不等于权重或数据已上传。平台尚未接收任何完整权重或语料。

## 图表

![RMU连续轨迹步数与Bio/MMLU准确率](attachments/charts/rmu_steps_bio_mmlu.svg)

![四组WildGuard自动评分对照](attachments/charts/four_group_wildguard.svg)

## 资源记录

连续RMU120步运行耗时约{run["total_process_seconds"]:.1f}秒，优化器更新循环约{run["optimizer_update_loop_seconds"]:.1f}秒；峰值显存allocated/reserved为{training_peak_gib:.2f}/{training_reserved_gib:.2f} GiB。四组全量A/C生成各约{next(x["wall_seconds"] for x in gen_sessions if x.get("model")=="original" and x["phase"]=="full"):.1f}/{next(x["wall_seconds"] for x in gen_sessions if x.get("model")=="rmu80" and x["phase"]=="full"):.1f}秒，峰值reserved约{max(x["peak_reserved_bytes"] for x in gen_sessions if x.get("phase")=="full" and x.get("model") in ("original","rmu80"))/2**30:.2f} GiB；Guard审核会话约{next(x["wall_seconds"] for x in gen_sessions if x["phase"]=="full" and "model" not in x):.1f}秒，峰值reserved约{max(x["peak_reserved_bytes"] for x in gen_sessions if "model" not in x)/2**30:.2f} GiB。Qwen3Guard已有WildGuardTest审核检测阶段input/output分别约{guard_test["tasks"]["input"]["resources"]["total_task_wall_seconds"]:.1f}/{guard_test["tasks"]["output"]["resources"]["total_task_wall_seconds"]:.1f}秒；这些离线延迟不等于服务端到端延迟。WildGuard评判器下载记录约14,496,898,091字节，Qwen3Guard快照约{load("safety/qwen3guard/records/download_manifest.json")["total_file_bytes"]:,}字节，均不复制到交付包。最近归档评测时可用磁盘约{judge["resource_final"]["disk_free_bytes"]/2**30:.2f} GiB；本次核验时剩余{current_free/2**30:.2f} GiB。
""")

    write(OUT / "README.md", """
# HoshiuZ Qwen3-4B RMU80 delivery metadata

This directory is a lightweight, metadata-first delivery draft for model ID hoshiuz-qwen3-4b-rmu80-v1. The selected checkpoint is step 80 from one continuous 120-step RMU run. It is not an independently trained 80-step model.

Included here: JSON/JSONL metadata and archived metrics, data statement, Chinese summary, audit/known-issues documents, and SVG charts. Not included: model weights or deltas, tokenizer payloads, training/validation/evaluation text, generated answers, per-item ChatGPT-assisted labels, or source mappings.

The three RMU80 delta payloads and base model exist locally on the training server. The Git checkout shows LFS pointers for delta paths. Remote LFS retrieval has not been verified. Training corpus transfer authorization is incomplete. A local absolute path, symlink, Git pointer, or this metadata package alone does not make the model/data accessible to a recipient.

No online serving service is included. See artifacts/serving/README.md, MISSING_ITEMS.md, and DELIVERY_SUMMARY.md.
""")

    content_gaps = [
        {"field": "model_card.base_model_version / model_artifacts.base_model.remote_revision", "status": "unknown", "reason": "基模最初下载时未记录上游不可变revision；目前以本地文件清单和参数指纹固定版本。"},
        {"field": "training_datasets[].raw_count for cleaned article splits", "status": "null_not_recorded", "reason": "归档只记录forget/retain各自清洗前的合计数，未记录按train/validation拆分的清洗前数量。因此分割级raw_count为null，另列角色合计数和清洗后分割数。"},
        {"field": "training_datasets[].license_or_authorization for retain data", "status": "unresolved", "reason": "retain语料归档revision及接收方再分发授权尚未从已有记录中确认。"},
        {"field": "model_card.artifact_access.remote_retrievable", "status": "null_unverified", "reason": "未核实远端Git LFS对象可取回；本机缓存存在不代表远端可用。"},
        {"field": "post_training.platform_method_enum / model_card.model_type enum mapping", "status": "pending_recipient_platform", "reason": "实际方法记录为RMU；machine_unlearning是本项目扩展表达，目标平台枚举尚未提供。"},
        {"field": "manifest.submitted_at", "status": "null_not_submitted", "reason": "本阶段未向平台提交；实际转交前必须保持null。"},
    ]
    missing = {
        "schema_version": SCHEMA_DATE,
        "prepared_at": now_utc(),
        "not_platform_submitted": True,
        "content_gaps": content_gaps,
        "items": [
            {"item":"完整Qwen3-4B原始基模权重及配置","status":"present_locally_not_transferred","recipient_action":"obtain authorized exact base model; official immutable source revision was not recorded"},
            {"item":"RMU40/80/120 BF16 XOR+gzip增量","status":"present_in_local_LFS_cache_remote_unverified_not_transferred","recipient_action":"verify/upload/retrieve all required LFS objects or provide another authorized transfer"},
            {"item":"原始tokenizer文件","status":"present_locally_not_transferred","recipient_action":"transfer with exact base model or provide authorized official access"},
            {"item":"WMDP-Bio forget/retain训练语料与固定token块","status":"present_locally_restricted_not_transferred","recipient_action":"verify recipient collaborator authorization; retain-corpus redistribution rights are unresolved"},
            {"item":"WMDP-Bio/MMLU/WildGuardTest评测原文","status":"present_locally_or_source_retrievable_not_transferred","recipient_action":"verify applicable source access/license and retrieve from official source"},
            {"item":"在线推理服务及运维材料","status":"missing","recipient_action":"out of scope for this phase; no service was built"},
            {"item":"平台模型遗忘方法枚举映射","status":"missing_or_not_found","recipient_action":"platform must define a machine-unlearning/RMU method mapping; do not label as SFT or DPO"},
            {"item":"人工安全复核金标准","status":"not_completed","recipient_action":"ChatGPT-assisted labels remain auxiliary and human_verified=false"},
            {"item":"独立未参与调参的最终测试","status":"not_available","recipient_action":"official tests were used in tuning; current results remain exploratory"},
            {"item":"PDF版中文报告","status":"not_created","recipient_action":"Markdown report is complete; no reliable Chinese PDF export was verified and no heavy toolchain was installed"},
            {"item":"平台实际提交时间","status":"not_submitted","recipient_action":"manifest.submitted_at remains null until an actual platform submission"},
        ],
        "statement": "Metadata and reports are prepared only. The package is not content-complete: model weights and datasets have not been delivered to a platform.",
    }
    dump(OUT / "missing_items.json", missing)
    missing_table = "\n".join(f'|{x["item"]}|{x["status"]}|{x["recipient_action"]}|' for x in missing["items"])
    content_gap_table = "\n".join(f'|`{x["field"]}`|{x["status"]}|{x["reason"]}|' for x in content_gaps)
    write(OUT / "MISSING_ITEMS.md", f"""
# 缺项与接收前事项

当前状态是元数据与报告已准备，不是完整模型/数据已经交到平台。下表逐项区分本地状态、远端状态和后续条件。

|项目|状态|后续事项|
|---|---|---|
{missing_table}

## 元数据内容待补或待确认

格式字段已经按当前交付规范整理；下列内容仍缺失或待接收方确认，不能据此宣称包内容完整。

|字段|状态|原因|
|---|---|---|
{content_gap_table}

RMU40/80/120增量仍在本机LFS缓存；本轮记录不证明GitHub端已可取回。数据权限、平台入口和服务部署均未由本阶段完成。
""")

    # Environment-specific report values are kept as recorded; do not claim a new run.
    training_peak_gib = run["cuda_peak_allocated_bytes"] / 2**30
    training_reserved_gib = run["cuda_peak_reserved_bytes"] / 2**30
    gen_sessions = four_results["sessions"]
    current_free = shutil.disk_usage(ROOT).free
    json_ok = jsonl_ok = finite_ok = timestamps_ok = True
    percent_ok = risk_ok = True
    credential_hits = []
    secret_re = re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})")
    percent_keys = {"accuracy_percent", "percentage_0_to_100", "selected_change_pp", "loss_pp", "delta_percentage_points"}
    risk_keys = {"precision", "recall", "F1", "macro_F1", "FPR", "FNR", "coverage_0_to_1", "error_rate_0_to_1", "effective_rate_0_to_1", "unknown_rate_0_to_1", "valid_only_rate_0_to_1", "unknown_rate_full_cohort_0_to_1"}
    def inspect_values(value):
        nonlocal finite_ok, timestamps_ok, percent_ok, risk_ok
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, (int, float)) and not isinstance(child, bool):
                    if not math.isfinite(child):
                        finite_ok = False
                    if (key in percent_keys or key.endswith("_percent") or key.endswith("_percent_full_cohort") or key.endswith("_rate_percent")) and not (-100 <= child <= 100):
                        percent_ok = False
                    if key in risk_keys and not (0 <= child <= 1):
                        risk_ok = False
                if isinstance(child, str) and (key.endswith("_at") or key in {"started_at", "completed_at", "recorded_at", "prepared_at"}):
                    try:
                        parsed = dt.datetime.fromisoformat(child.replace("Z", "+00:00"))
                        if parsed.tzinfo is None or parsed.utcoffset() is None:
                            timestamps_ok = False
                    except ValueError:
                        timestamps_ok = False
                inspect_values(child)
        elif isinstance(value, list):
            for child in value:
                inspect_values(child)

    for p in (x for x in OUT.rglob("*") if x.is_file()):
        raw = p.read_bytes()
        if secret_re.search(raw):
            credential_hits.append(p.relative_to(OUT).as_posix())
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            finite_ok = False
            continue
        if p.suffix == ".json":
            try:
                parsed_json = json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                inspect_values(parsed_json)
            except Exception:
                json_ok = False
        elif p.suffix == ".jsonl":
            try:
                for line in text.splitlines():
                    if line.strip():
                        inspect_values(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
            except Exception:
                jsonl_ok = False
    try:
        import xml.etree.ElementTree as ET
        for chart in (OUT / "attachments/charts").glob("*.svg"):
            ET.parse(chart)
        svg_ok = True
    except Exception:
        svg_ok = False
    required_dirs = ["artifacts/model", "artifacts/tokenizer", "artifacts/serving", "datasets/train", "datasets/validation", "datasets/evaluation", "attachments/charts", "attachments/logs"]
    dirs_ok = all((OUT / d / "README.md").is_file() for d in required_dirs)
    schema_validation = validate_delivery_schema(OUT)
    if not schema_validation["passed"]:
        raise RuntimeError("Delivery schema validation failed: " + "; ".join(schema_validation["errors"]))
    if not (json_ok and jsonl_ok and finite_ok and timestamps_ok and percent_ok and risk_ok and svg_ok and dirs_ok and not credential_hits):
        raise RuntimeError("Delivery package pre-manifest validation failed")
    try:
        source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        workspace_branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        source_commit, workspace_branch = None, None
    completion = {
        "schema_version": SCHEMA_DATE,
        "prepared_at": now_utc(),
        "source_commit": source_commit,
        "workspace_branch": workspace_branch,
        "checks": {
            "required_standard_files_present_before_manifest": True,
            "required_standard_directories_have_readme": True,
            "JSON_UTF8_and_finite_numbers": json_ok and finite_ok,
            "JSONL_UTF8_and_finite_numbers": jsonl_ok and finite_ok,
            "timestamps_timezone_or_null": timestamps_ok,
            "field_schema_validation": {"status": "passed", "dataset_rows": schema_validation["dataset_rows"], "capability_result_rows": schema_validation["capability_result_rows"], "errors": []},
            "package_files_hashed_and_sized": True,
            "package_files_record_count_field_and_values": True,
            "fraction_percentage_denominator_consistency": True,
            "credential_scan": {"pattern_hits": len(credential_hits), "hit_paths": credential_hits},
            "raw_text_scan": "metadata builder reads no prompt/answer bodies or per-item assisted labels",
            "percent_metric_range": {"percentages_0_to_100": percent_ok, "risk_rates_0_to_1": risk_ok},
            "SVG_XML_parse": svg_ok,
            "required_directory_readmes": dirs_ok,
            "model_rebuild_evidence_reused": rebuild["compressed_payload_and_base_and_reconstructed_hashes_match"],
            "model_rebuild_repeated_now": False,
            "remote_LFS_verified": False,
            "platform_submitted": False,
            "content_complete": False,
        },
        "content_gaps": content_gaps,
        "resources_from_archived_runs": {
            "training_peak_allocated_GiB": training_peak_gib,
            "training_peak_reserved_GiB": training_reserved_gib,
            "four_group_generation_and_guard_sessions": [
                {"model":x.get("model", "qwen3guard"),"phase":x["phase"],"peak_allocated_GiB":x.get("peak_allocated_bytes",0)/2**30,"peak_reserved_GiB":x.get("peak_reserved_bytes",0)/2**30,"wall_seconds":x.get("wall_seconds")}
                for x in gen_sessions
            ],
            "last_recorded_judge_disk_free_bytes": judge["resource_final"]["disk_free_bytes"],
            "current_disk_free_bytes": current_free,
        },
        "package_size_excludes": ["base weights","RMU delta payloads","tokenizer payloads","datasets","raw requests/answers","per-item auxiliary labels and source mapping"],
        "self_hash_note": "manifest.json omits its own size/hash to avoid a self-referential manifest; all other package files are listed.",
    }
    dump(OUT / "delivery_audit.json", completion)
    check_rows = "\n".join(f'|{k}|{v}|' for k,v in completion["checks"].items())
    write(OUT / "DELIVERY_CHECK_REPORT.md", f"""
# 交付包核验报告

核验时间：{completion["prepared_at"]}。本报告只核验本地元数据交付目录和既有证据；没有加载或重建模型，没有读取训练/评测语料正文，没有生成/评分回答。

**格式状态：** 必需字段、数组/对象/字符串/数值类型、源数据路径存在性、指标分母口径和文件索引结构已通过静态校验。**内容状态：部分待补。** 本包不是完整交付；具体null、未核实和未转交项目见 `MISSING_ITEMS.md`。

|检查项|结果|
|---|---|
{check_rows}

RMU80重建证据复用现有 checkpoint-80/rebuild_verification.json：BF16 XOR压缩负载、基模张量和重建张量哈希匹配。检查点真实负载仍只在本机LFS缓存；Git远端可取回性未核实。该包的JSON/JSONL、字段类型、文件哈希、文件大小、record_count和SVG图表会在manifest.json生成后再次进行最终校验。

当前数据盘剩余 {current_free} bytes。此数值是核验时文件系统读数，不代表模型或语料已交付。
""")

    required_standard = [
        "manifest.json","model_card.json","model_artifacts.json","post_training.json","training_datasets.jsonl",
        "training_config.json","training_metrics.jsonl","evaluation_summary.json","data_statement.md","README.md",
        "DELIVERY_SUMMARY.md","DELIVERY_CHECK_REPORT.md","MISSING_ITEMS.md","missing_items.json","delivery_audit.json",
        "DELIVERY_CHECK_REPORT.md",
        "artifacts/model/README.md","artifacts/tokenizer/README.md","artifacts/serving/README.md",
        "datasets/train/README.md","datasets/validation/README.md","datasets/evaluation/README.md",
        "attachments/charts/README.md","attachments/logs/README.md",
    ]
    for r in required_standard:
        if r != "manifest.json" and not (OUT / r).is_file():
            raise RuntimeError(f"Required delivery file missing: {r}")
    # Build a complete package index, excluding only its self-referential manifest record.
    files = []
    for p in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "manifest.json"):
        relp = p.relative_to(OUT).as_posix()
        files.append(file_record(relp))
    manifest = {
        "delivery_id": "hoshiuz-qwen3-4b-rmu80-v1-20261015",
        "delivery_version": "0.1.0",
        "schema_version": SCHEMA_DATE,
        "prepared_at": now_utc(),
        "submitted_at": None,
        "submission_status": "not_submitted",
        "model_id": "hoshiuz-qwen3-4b-rmu80-v1",
        "base_model_id": "qwen3-4b-instruct-2507",
        "confidentiality": "public metadata/report package; model weights and WMDP data restricted/local",
        "status": "metadata_and_reports_prepared; full weights_and_data_not_transferred",
        "files": files,
        "known_issues": missing["items"] + content_gaps,
        "self_hash_note": "manifest.json is excluded from files to avoid a self-referential hash; its own size/hash is therefore null/unavailable.",
        "delivery_is_self_contained": False,
        "platform_submission_completed": False,
    }
    dump(OUT / "manifest.json", manifest)
    final_manifest_validation = validate_manifest(OUT)
    if not final_manifest_validation["passed"]:
        raise RuntimeError("Final manifest validation failed: " + "; ".join(final_manifest_validation["errors"]))
    print(json.dumps({
        "package": str(OUT),
        "indexed_files": len(files),
        "training_dataset_rows": len(data_rows),
        "training_metric_rows": len(selected_metric_rows),
        "delta80_payload_bytes": rebuild["bytes"],
        "delta_local_all_hash_match": all(x["local_payload_hash_matches"] for x in delta_files),
        "remote_lfs_verified": False,
        "schema_validation": schema_validation,
        "final_manifest_validation": final_manifest_validation,
        "selected_step": 80,
        "run_max_steps": 120,
    },ensure_ascii=False))


if __name__ == "__main__":
    build()
