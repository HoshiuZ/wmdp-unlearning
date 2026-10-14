"""Preflight checks safe to run before model-file access is granted."""
import json
from pathlib import Path

from .common import CFG, HERE, LOCAL, MODEL_SNAPSHOT, ROOT, OFFICIAL_TEMPLATE, digest, dump, read_jsonl, sha


def main():
    manifest = json.loads((LOCAL / "manifest.json").read_text())
    inputs = read_jsonl(LOCAL / "scoring_inputs.jsonl")
    labels = read_jsonl(LOCAL / "validation_labels.jsonl")
    blind = read_jsonl(LOCAL / "blind_mapping.jsonl")
    assert len(inputs) == 5588 and len({x["content_sha256"] for x in inputs}) == len(inputs)
    assert all(set(x) == {"content_sha256", "prompt", "response"} for x in inputs)
    assert all(x["content_sha256"] == digest([x["prompt"], x["response"]]) for x in inputs)
    assert len(blind) == 10194
    assert len(labels) == 1722
    assert manifest["four_groups"]["source_generation_rows_verified"] == 3398
    assert manifest["four_groups"]["fixed_refusals_verified"] + manifest["four_groups"]["protected_raw_answers_verified"] == 6796
    assert manifest["wildguardtest"]["harm_labeled_rows"] == 1709
    assert manifest["wildguardtest"]["refusal_labeled_rows"] == 1720
    assert manifest["wildguardtest"]["file_sha256"] == "6ccc2909c1ae6d41424fac69f1fc32535b1de39cb8f80407d81e8bc64a0bebca"
    sample = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
    from .common import parse_official_output
    parsed = parse_official_output(sample)
    assert parsed["fields"] == {"prompt_harmfulness": True, "response_refusal": True, "response_harmfulness": False}
    assert not parsed["is_parsing_error"]
    downloaded = MODEL_SNAPSHOT.exists()
    receipt_path = HERE / "model_source.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if downloaded:
        assert receipt.get("revision") == CFG["revision"]
        assert not any(MODEL_SNAPSHOT.glob("pytorch_model*.bin")), "Duplicate full-precision .bin shards should not be cached"
    report = {
        "model_id": CFG["model_id"], "revision": CFG["revision"], "license": CFG["license"],
        "official_file_access": "snapshot_available" if downloaded else "blocked_http_403_not_in_authorized_list",
        "model_weights_downloaded_bytes": receipt.get("safetensors_bytes", 0) if downloaded else 0,
        "model_snapshot_created": downloaded,
        "safetensors_expected_bytes_from_official_metadata": 14496097336,
        "protected_mapping_rows_verified": len(blind), "raw_answer_rows_verified": 3398,
        "fixed_refusal_rows_verified": manifest["four_groups"]["fixed_refusals_verified"],
        "protected_raw_answer_rows_verified": manifest["four_groups"]["protected_raw_answers_verified"],
        "wildguardtest_rows": manifest["wildguardtest"]["raw_rows"],
        "wildguardtest_response_harm_labels": manifest["wildguardtest"]["harm_labeled_rows"],
        "wildguardtest_response_refusal_labels": manifest["wildguardtest"]["refusal_labeled_rows"],
        "unique_scoring_pairs_prepared": len(inputs), "official_template_parser_fixture": "passed",
        "request_and_answer_only_in_inference_inputs": True,
        "gated_entry_url": CFG["official_model_card"],
        "official_template_sha256": sha(OFFICIAL_TEMPLATE.encode()),
        "input_file_sha256": sha((LOCAL / "scoring_inputs.jsonl").read_bytes()),
        "validation_labels_sha256": sha((LOCAL / "validation_labels.jsonl").read_bytes()),
        "model_weights_cache_path": str((ROOT / "cache/huggingface/hub/models--allenai--wildguard").relative_to(ROOT)),
    }
    dump(HERE / "preflight.json", report)
    print(json.dumps({"preflight": "snapshot_available" if downloaded else "blocked_model_files",
                      "pairs": len(inputs), "mapping_rows": len(blind),
                      "official_parser_fixture": "passed",
                      "model_weights_downloaded_bytes": report["model_weights_downloaded_bytes"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
