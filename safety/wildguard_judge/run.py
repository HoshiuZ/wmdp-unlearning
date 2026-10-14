"""Score original WildGuardTest pairs and the already-generated blind responses."""
import argparse
import json
import platform
import sqlite3
import subprocess
import time
from pathlib import Path

from .common import (CFG, HERE, LOCAL, MODEL_SNAPSHOT, ROOT, OFFICIAL_TEMPLATE, digest, dump, open_db,
                    official_prompt, parse_official_output, read_jsonl, resource_snapshot,
                    score_key, sha)


def load_scorer():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError("Pinned WildGuard snapshot is not downloaded; run python -m safety.wildguard_judge.download")
    free_vram, total_vram = torch.cuda.mem_get_info(CFG["generation"]["device"])
    # The two official BF16 shards are ~13.5 GiB. Keep headroom for activations and cache.
    if free_vram < 18 * 2**30:
        raise RuntimeError(f"Insufficient free VRAM for one BF16 judge: {free_vram} bytes free")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_SNAPSHOT), use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_SNAPSHOT), dtype=torch.bfloat16,
        attn_implementation=CFG["generation"]["attention_implementation"],
        low_cpu_mem_usage=True, local_files_only=True,
    ).to(CFG["generation"]["device"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Judge model unexpectedly has trainable parameters")
    model_context = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    tokenizer_context = int(getattr(tokenizer, "model_max_length", 10**12) or 10**12)
    context_limit = min(model_context, tokenizer_context)
    if context_limit <= 0 or context_limit > 10**9:
        context_limit = model_context
    if not context_limit:
        raise RuntimeError("Could not determine the model context limit")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Tokenizer has no EOS token")
    runtime = {
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "model_config_context": model_context,
        "tokenizer_max_length": tokenizer_context, "effective_context_limit": context_limit,
        "parameter_count": int(model.num_parameters()), "dtype": "bfloat16", "device": CFG["generation"]["device"],
        "attention_implementation": CFG["generation"]["attention_implementation"],
        "eos_token_id": eos, "pad_token_id": tokenizer.pad_token_id,
        "input_template_sha256": sha(OFFICIAL_TEMPLATE.encode()),
        "cuda_free_before_load_bytes": int(free_vram), "cuda_total_bytes": int(total_vram),
        "model_id": CFG["model_id"], "revision": CFG["revision"],
    }
    fingerprint = digest({"runtime": runtime, "config": CFG["generation"], "revision": CFG["revision"],
                          "template_sha256": runtime["input_template_sha256"]})
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - t0
    return torch, tokenizer, model, runtime, fingerprint, load_seconds


def score_one(prompt, response, tokenizer, model, runtime, torch):
    rendered = official_prompt(prompt, response)
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False)
    input_tokens = int(ids["input_ids"].shape[-1])
    content_sha = digest([prompt, response])
    base = {"content_sha256": content_sha, "input_tokens": input_tokens,
            "label_source": "wildguard_auto",
            "input_text_sha256": sha(rendered.encode()), "generation_config": CFG["generation"],
            "fields": {"prompt_harmfulness": None, "response_refusal": None, "response_harmfulness": None},
            "field_errors": {}, "is_parsing_error": True, "raw_output": "", "generated_tokens": 0,
            "finish_reason": "not_started", "status": "unknown"}
    if input_tokens + int(CFG["generation"]["max_new_tokens"]) > runtime["effective_context_limit"]:
        base["error_code"] = "context_limit_exceeded"
        return base
    device = CFG["generation"]["device"]
    ids = {k: v.to(device) for k, v in ids.items()}
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **ids,
                do_sample=False,
                num_beams=1,
                max_new_tokens=int(CFG["generation"]["max_new_tokens"]),
                use_cache=True,
                eos_token_id=runtime["eos_token_id"],
                pad_token_id=runtime["eos_token_id"],
            )
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        new_ids = output_ids[0, input_tokens:].detach().cpu()
        generated_tokens = int(new_ids.numel())
        raw = tokenizer.decode(new_ids, skip_special_tokens=True)
        parsed = parse_official_output(raw)
        eos_id = runtime["eos_token_id"]
        if generated_tokens and int(new_ids[-1]) == eos_id:
            finish = "eos"
        elif generated_tokens >= int(CFG["generation"]["max_new_tokens"]):
            finish = "max_new_tokens"
        else:
            finish = "stopped_without_eos"
        base.update(parsed)
        base.update(raw_output=raw, generated_tokens=generated_tokens, finish_reason=finish,
                    generation_seconds=time.perf_counter() - started)
        if finish != "eos":
            base["status"] = "unknown"
            base["error_code"] = "generation_incomplete_or_token_limit"
            base["fields"] = {k: None for k in base["fields"]}
            base["field_errors"] = {k: "generation_incomplete_or_token_limit" for k in base["fields"]}
            base["is_parsing_error"] = True
        elif parsed["is_parsing_error"]:
            base["status"] = "parse_error"
            base["error_code"] = "official_output_parse_error"
        else:
            base["status"] = "ok"
            base["error_code"] = None
        return base
    except Exception as exc:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        base.update(status="error", error_code=type(exc).__name__,
                    generation_seconds=time.perf_counter() - started,
                    field_errors={k: "inference_exception" for k in base["fields"]})
        return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    inputs = read_jsonl(LOCAL / "scoring_inputs.jsonl")
    labels = read_jsonl(LOCAL / "validation_labels.jsonl")
    validation_order = []
    seen = set()
    for item in labels:
        if item["content_sha256"] not in seen:
            validation_order.append(item["content_sha256"])
            seen.add(item["content_sha256"])
    by_sha = {x["content_sha256"]: x for x in inputs}
    if args.phase == "pilot":
        selected = [by_sha[s] for s in validation_order[:8]]
    else:
        selected = inputs
    if not selected:
        raise RuntimeError("No scoring inputs prepared")

    torch, tokenizer, model, runtime, fingerprint, load_seconds = load_scorer()
    torch.cuda.reset_peak_memory_stats()
    conn = open_db()
    session_path = LOCAL / "sessions" / f"{args.phase}_{time.time_ns()}.json"
    initial = resource_snapshot()
    started = time.perf_counter()
    session = {"phase": args.phase, "started_unix": time.time(), "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "model_load_seconds": load_seconds, "runtime": runtime, "scorer_fingerprint": fingerprint,
        "selected_pairs": len(selected), "inference_new": 0, "cache_reused": 0, "unknown_or_error": 0,
        "elapsed_inference_seconds": 0, "initial_resources": initial,
        "source_sha256": {name: sha((HERE / name).read_bytes()) for name in ("common.py", "run.py", "config.json")}}
    consecutive = 0
    for index, item in enumerate(selected):
        content_sha = item["content_sha256"]
        key = score_key(content_sha, fingerprint)
        record = conn.execute("SELECT result FROM scores WHERE key=?", (key,)).fetchone()
        if record is not None:
            session["cache_reused"] += 1
        else:
            # Classifier input is exactly this pair; no group, labels, guard output, or target labels enter.
            result = score_one(item["prompt"], item["response"], tokenizer, model, runtime, torch)
            conn.execute("INSERT INTO scores VALUES (?,?,?)", (key, content_sha, json.dumps(result, ensure_ascii=False)))
            conn.commit()
            session["inference_new"] += 1
            consecutive = consecutive + 1 if result["status"] != "ok" else 0
            session["unknown_or_error"] += result["status"] != "ok"
            if consecutive >= CFG["stop_after_consecutive_errors"]:
                raise RuntimeError("Five consecutive inference/parse errors; stopping with per-item cache intact")
        session.update(elapsed_inference_seconds=time.perf_counter() - started,
                       peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                       peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                       after_resources=resource_snapshot())
        dump(session_path, session)
        if (index + 1) % 100 == 0 or index + 1 == len(selected):
            print(json.dumps({"phase": args.phase, "completed": index + 1, "selected": len(selected),
                              "new": session["inference_new"], "reused": session["cache_reused"],
                              "errors": session["unknown_or_error"]}), flush=True)
    session["complete"] = True
    session["wall_seconds"] = time.perf_counter() - started
    dump(session_path, session)
    dump(LOCAL / "runtime.json", {"runtime": runtime, "fingerprint": fingerprint})
    if args.phase == "pilot":
        rows = []
        for item in selected:
            result = json.loads(conn.execute("SELECT result FROM scores WHERE key=?",
                                             (score_key(item["content_sha256"], fingerprint),)).fetchone()[0])
            rows.append({"status": result["status"], "is_parsing_error": result["is_parsing_error"],
                         "field_errors": result["field_errors"], "finish_reason": result["finish_reason"],
                         "input_tokens": result["input_tokens"], "generated_tokens": result["generated_tokens"],
                         "prompt_harmfulness_present": result["fields"]["prompt_harmfulness"] is not None,
                         "response_refusal_present": result["fields"]["response_refusal"] is not None,
                         "response_harmfulness_present": result["fields"]["response_harmfulness"] is not None})
        dump(HERE / "pilot_check.json", {"n": len(rows), "records": rows,
                                         "all_three_fields_parsed": all(all(x[k] for k in (
                                             "prompt_harmfulness_present", "response_refusal_present",
                                             "response_harmfulness_present")) for x in rows),
                                         "all_status_ok": all(x["status"] == "ok" for x in rows),
                                         "no_truncation": all(x["finish_reason"] == "eos" for x in rows)})
        if not (HERE / "pilot_check.json").exists():
            raise RuntimeError("Pilot output verification was not written")
        print("PILOT_SAVED", str(HERE / "pilot_check.json"), flush=True)
    else:
        print("FULL_SCORING_COMPLETE", session["inference_new"], session["cache_reused"], flush=True)


if __name__ == "__main__":
    main()
