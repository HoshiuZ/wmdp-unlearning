"""Validate the existing blind mapping and prepare private judge inputs."""
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .common import CFG, HERE, LOCAL, ROOT, digest, dump, read_jsonl, sha, write_jsonl


def main():
    import pyarrow.parquet as pq

    four_cfg = json.loads((ROOT / CFG["four_group_config"]).read_text())
    four_manifest = json.loads((ROOT / CFG["four_group_manifest"]).read_text())
    old_root = ROOT / four_cfg["local_root"]
    requests = read_jsonl(old_root / "requests.jsonl")
    review = read_jsonl(ROOT / CFG["blind_review"])
    mappings = read_jsonl(ROOT / CFG["blind_mapping"])
    systems = read_jsonl(ROOT / CFG["systems"])
    request_by_id = {row["sample_id"]: row for row in requests}
    review_by_id = {row["blind_id"]: row for row in review}
    review_by_content = {digest([row["request"], row["answer"]]): row for row in review}
    if len(request_by_id) != 1699 or len(review_by_id) != len(review):
        raise RuntimeError("Existing four-group request/review inventory is incomplete or duplicated")
    if len(review_by_content) != len(review):
        raise RuntimeError("Existing review material contains duplicate or colliding answer pairs")
    if four_manifest["source_input_manifest_sha256"] != "ad4f30e1d9e6eb1094c6d13187a900f55589eda2865e4687f42b915d7bacd8d2":
        raise RuntimeError("Four-group request manifest is not the previously fixed 1699-request set")
    if len(mappings) != 6 * len(requests):
        raise RuntimeError(f"Expected 6 mapping rows per request; found {len(mappings)}")

    # Verify each blind answer against the original generation cache or the fixed returned refusal.
    generation_db = sqlite3.connect(ROOT / CFG["four_group_records"])
    systems_by = {(r["sample_id"], r["group"], r["policy"]): r for r in systems}
    qwen_db = generation_db
    checked_raw = checked_refusal = checked_protected_raw = 0
    qwen_grade_by_content = {}
    for row in systems:
        if row["output_audit_ref"]:
            rec = qwen_db.execute("SELECT task,sample,result FROM records WHERE key=?", (row["output_audit_ref"],)).fetchone()
            if rec is None or rec[0] != "output_audit" or rec[1] != row["sample_id"]:
                raise RuntimeError("Output audit reference is absent or points to another task/sample")
            audit = json.loads(rec[2])
            req = request_by_id[row["sample_id"]]["request"]
            raw_generation = generation_db.execute("SELECT result FROM records WHERE key=?", (row["generation_ref"],)).fetchone()
            if raw_generation is None:
                raise RuntimeError("Qwen3Guard output audit has no source generation")
            raw_answer = json.loads(raw_generation[0]).get("response")
            qwen_content = digest([req, raw_answer])
            grade = audit.get("label") if audit.get("status") == "ok" and audit.get("parse_ok") else None
            qwen_grade_by_content[qwen_content] = grade
    for item in mappings:
        source_request = request_by_id.get(item["sample_id"])
        blind = review_by_id.get(item["blind_id"])
        if source_request is None or blind is None:
            raise RuntimeError("Blind mapping references an unknown request or answer")
        if source_request["request"] != blind["request"]:
            raise RuntimeError("Blind mapping request text does not match the fixed request list")
        pair_sha = digest([blind["request"], blind["answer"]])
        if pair_sha != item["content_key"]:
            raise RuntimeError("Blind mapping content hash mismatch")
        if item["group"] in ("A", "C"):
            if item["policy"] is not None or item["state"] != "raw_answer":
                raise RuntimeError("Raw A/C mapping has unexpected policy/state")
            expected_model = "original" if item["group"] == "A" else "rmu80"
            gen = generation_db.execute("SELECT task,sample,result FROM records WHERE key=?", (item["generation_ref"],)).fetchone()
            if gen is None or gen[0] != "generation" or gen[1] != item["sample_id"]:
                raise RuntimeError("A/C generation reference is absent or misaligned")
            generated = json.loads(gen[2])
            if generated.get("status") != "ok" or generated.get("model") != expected_model:
                raise RuntimeError("A/C source generation is failed or belongs to the wrong model")
            if generated.get("response_sha256") != sha(blind["answer"].encode()):
                raise RuntimeError("A/C blind answer differs from its cached raw model answer")
            checked_raw += 1
        elif item["group"] in ("B", "D"):
            if item["policy"] not in four_cfg["policies"]:
                raise RuntimeError("Protected group mapping has an unknown policy")
            sysrow = systems_by.get((item["sample_id"], item["group"], item["policy"]))
            if sysrow is None or sysrow["state"] != item["state"] or sysrow["generation_ref"] != item["generation_ref"]:
                raise RuntimeError("Protected response mapping does not match system record")
            if sysrow["input_label"] != source_request["gold"]:
                raise RuntimeError("Input-label grouping metadata is inconsistent")
            gen = generation_db.execute("SELECT task,sample,result FROM records WHERE key=?", (item["generation_ref"],)).fetchone()
            if gen is None or gen[0] != "generation" or gen[1] != item["sample_id"]:
                raise RuntimeError("Protected response generation reference is absent")
            generated = json.loads(gen[2])
            if item["state"] == "allow":
                if generated.get("status") != "ok" or generated.get("response_sha256") != sha(blind["answer"].encode()):
                    raise RuntimeError("Allowed protected answer differs from the cached generated response")
                checked_protected_raw += 1
            else:
                if blind["answer"] != four_cfg["fixed_refusal"]:
                    raise RuntimeError("Blocked/error protected output is not the fixed refusal")
                checked_refusal += 1
        else:
            raise RuntimeError("Unexpected group in blind mapping")
    if checked_raw != 2 * len(requests) or checked_refusal + checked_protected_raw != 4 * len(requests):
        raise RuntimeError("Not all six response variants were mapped")

    # The model sees only prompt+response. Labels, model/group names, and Guard judgments remain separate.
    blind_inputs = [{"content_sha256": digest([r["request"], r["answer"]]),
                     "prompt": r["request"], "response": r["answer"]} for r in review]
    parquet = ROOT / CFG["dataset_parquet_snapshot"]
    raw_rows = pq.read_table(parquet).to_pylist()
    if len(raw_rows) != 1725:
        raise RuntimeError(f"Expected 1725 fixed WildGuardTest rows, got {len(raw_rows)}")
    validation_inputs = []
    validation_labels = []
    label_counts = {"response_harm_label": Counter(), "response_refusal_label": Counter()}
    for row_index, row in enumerate(raw_rows):
        prompt = row.get("prompt")
        response = row.get("response")
        harm = row.get("response_harm_label")
        refusal = row.get("response_refusal_label")
        if harm is not None:
            label_counts["response_harm_label"][str(harm)] += 1
        if refusal is not None:
            label_counts["response_refusal_label"][str(refusal)] += 1
        valid_harm = harm in ("harmful", "unharmful")
        valid_refusal = refusal in ("refusal", "compliance")
        if not prompt or not response or not (valid_harm or valid_refusal):
            continue
        content_sha = digest([prompt, response])
        validation_inputs.append({"content_sha256": content_sha, "prompt": prompt, "response": response})
        validation_labels.append({"row_index": row_index, "content_sha256": content_sha,
                                  "response_harm_gold": harm if valid_harm else None,
                                  "response_refusal_gold": refusal if valid_refusal else None})
    # Score an exact request/answer pair once across validation and new generations.
    all_inputs = {}
    for item in validation_inputs + blind_inputs:
        prior = all_inputs.get(item["content_sha256"])
        if prior is not None and (prior["prompt"], prior["response"]) != (item["prompt"], item["response"]):
            raise RuntimeError("SHA-256 collision while combining evaluator inputs")
        all_inputs[item["content_sha256"]] = item
    combined = [all_inputs[key] for key in sorted(all_inputs)]

    LOCAL.mkdir(parents=True, exist_ok=True)
    write_jsonl(LOCAL / "scoring_inputs.jsonl", combined)
    write_jsonl(LOCAL / "validation_labels.jsonl", validation_labels)
    write_jsonl(LOCAL / "blind_mapping.jsonl", mappings)
    write_jsonl(LOCAL / "evaluation_requests.jsonl", [
        {"sample_id": r["sample_id"], "input_label": r["gold"]} for r in requests
    ])
    write_jsonl(LOCAL / "qwen_guard_pair_grades.jsonl", [
        {"content_sha256": content_sha, "label": label}
        for content_sha, label in sorted(qwen_grade_by_content.items())
    ])
    data_metadata_path = ROOT / "safety/qwen3guard/wildguardtest/data_metadata.json"
    parquet_sha = sha(parquet.read_bytes())
    manifest = {
        "protocol": CFG["version"], "created_without_label_text_in_inference_inputs": True,
        "model_id": CFG["model_id"], "revision": CFG["revision"],
        "generation": CFG["generation"], "official_template_sha256": sha((__import__("safety.wildguard_judge.common", fromlist=["OFFICIAL_TEMPLATE"]).OFFICIAL_TEMPLATE).encode()),
        "wildguardtest": {"dataset_id": CFG["dataset_id"], "revision": CFG["dataset_revision"],
                          "config": CFG["dataset_config"], "split": CFG["dataset_split"],
                          "file": CFG["dataset_file"], "raw_rows": len(raw_rows),
                          "file_bytes": parquet.stat().st_size, "file_sha256": parquet_sha,
                          "non_null_labels": {k: dict(v) for k, v in label_counts.items()},
                          "harm_labeled_rows": sum(x["response_harm_gold"] is not None for x in validation_labels),
                          "refusal_labeled_rows": sum(x["response_refusal_gold"] is not None for x in validation_labels),
                          "unique_scored_pairs": len({x["content_sha256"] for x in validation_inputs}),
                          "validation_mapping_sha256": sha((LOCAL / "validation_labels.jsonl").read_bytes())},
        "four_groups": {"source_commit": four_manifest["code_commit_before_run"],
                        "requests": len(requests), "input_labels": dict(Counter(r["gold"] for r in requests)),
                        "source_generation_rows_verified": checked_raw,
                        "protected_raw_answers_verified": checked_protected_raw,
                        "fixed_refusals_verified": checked_refusal,
                        "unique_blind_request_answer_pairs": len(review),
                        "blind_mappings_verified": len(mappings),
                        "request_manifest_sha256": four_manifest["source_input_manifest_sha256"],
                        "blind_file_sha256": sha((ROOT / CFG["blind_review"]).read_bytes()),
                        "mapping_file_sha256": sha((LOCAL / "blind_mapping.jsonl").read_bytes())},
        "scoring_input_pairs_union": len(combined),
        "scoring_input_sha256": sha((LOCAL / "scoring_inputs.jsonl").read_bytes()),
        "all_generation_inputs_contain_only_pair_hash_and_prompt_response": True,
        "resource_before_download_or_model_load": {"disk_free_bytes": __import__("shutil").disk_usage(ROOT).free}
    }
    dump(LOCAL / "manifest.json", manifest)
    dump(HERE / "data_manifest.json", {k: v for k, v in manifest.items() if k not in ("all_generation_inputs_contain_only_pair_hash_and_prompt_response",)})
    print(json.dumps({"prepared": True, "blind_pairs": len(review), "mapping_rows": len(mappings),
                      "verified_raw_answers": checked_raw, "verified_protected_answers": checked_protected_raw,
                      "verified_fixed_refusals": checked_refusal, "dataset_rows": len(raw_rows),
                      "harm_label_rows": manifest["wildguardtest"]["harm_labeled_rows"],
                      "refusal_label_rows": manifest["wildguardtest"]["refusal_labeled_rows"],
                      "combined_unique_pairs": len(combined), "parquet_sha256": parquet_sha}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
