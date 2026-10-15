#!/usr/bin/env python3
"""Compare the pinned base model and RMU80 interactively, one prompt at a time."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/Qwen3-4B-Instruct-2507"
DELTA = (
    ROOT
    / "open-unlearning/saves/unlearn/rmu_step_comparison_v1/"
    / "h12_a2_s32_seed42_continuous120/checkpoint-80"
)
FOUR_GROUPS = ROOT / "safety/four_groups"
LOCAL = ROOT / "cache/four_groups_v1"


def digest_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_full_fingerprints(model, tensor_hash, expected_path: Path, expected_digest: str, label: str) -> dict:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    parameters = list(model.named_parameters())
    actual = {}
    for index, (name, parameter) in enumerate(parameters, start=1):
        actual[name] = tensor_hash(parameter)
        if index % 40 == 0 or index == len(parameters):
            print(f"{label}指纹校验：{index}/{len(parameters)}", flush=True)

    if actual != expected:
        mismatches = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError(
            f"{label}完整参数指纹不匹配：{', '.join(mismatches[:3]) or '未知参数'}"
        )
    actual_digest = digest_json(actual)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"{label}指纹摘要不匹配：期望 {expected_digest}，实际 {actual_digest}"
        )
    return expected


def load_compare_model():
    if not BASE.is_dir():
        raise FileNotFoundError(f"找不到基模目录：{BASE}")
    if not DELTA.is_dir():
        raise FileNotFoundError(f"找不到指定RMU80检查点：{DELTA}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA设备；本脚本要求使用本地GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持BF16，已停止加载")

    protocol = json.loads((FOUR_GROUPS / "manifest.json").read_text(encoding="utf-8"))
    base_fingerprints = LOCAL / "base_fingerprints.json"
    rmu_fingerprints = LOCAL / "rmu80_fingerprints.json"
    delta_manifest_path = DELTA / "checkpoint.json"
    for path in (base_fingerprints, rmu_fingerprints, delta_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少校验所需文件：{path}")
    delta_manifest_sha = hashlib.sha256(delta_manifest_path.read_bytes()).hexdigest()
    if delta_manifest_sha != protocol["delta_manifest_sha256"]:
        raise RuntimeError("指定checkpoint-80的增量清单与已验证实验清单不一致")

    print("正在读取本地tokenizer…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.padding_side = "left"
    template = tokenizer.chat_template
    template_sha = (
        hashlib.sha256(template.encode("utf-8")).hexdigest()
        if isinstance(template, str)
        else None
    )
    if template_sha != protocol["chat_template_sha256"]:
        raise RuntimeError("本地tokenizer聊天模板与已验证实验记录不一致")

    print("正在从本地加载单个基模实例（BF16，SDPA）…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.eval()
    model.requires_grad_(False)

    sys.path.insert(0, str(ROOT / "scripts"))
    from rmu_adaptation_utils import load_delta, tensor_hash

    print("校验原模型完整指纹…", flush=True)
    base_fp = verify_full_fingerprints(
        model,
        tensor_hash,
        base_fingerprints,
        protocol["base_fingerprint"],
        "原模型",
    )

    delta_manifest = json.loads(delta_manifest_path.read_text(encoding="utf-8"))
    if delta_manifest.get("format") != "bf16-uint16-xor-gzip-v1":
        raise RuntimeError("RMU增量格式与预期不符")
    update_rows = delta_manifest.get("parameters", [])
    if not update_rows:
        raise RuntimeError("RMU80增量中没有可切换参数")
    parameters = dict(model.named_parameters())
    update_names = {row["name"] for row in update_rows}
    if len(update_names) != len(update_rows) or not update_names.issubset(parameters):
        raise RuntimeError("RMU80增量参数名称重复或不属于当前基模")
    base_cache = {name: parameters[name].detach().clone() for name in update_names}

    print("应用指定的连续120步轨迹checkpoint-80增量…", flush=True)
    loaded_manifest = load_delta(model, DELTA)
    if loaded_manifest != delta_manifest:
        raise RuntimeError("RMU加载器读取的增量清单与预校验内容不一致")
    print("校验RMU80完整指纹…", flush=True)
    rmu_fp = verify_full_fingerprints(
        model,
        tensor_hash,
        rmu_fingerprints,
        protocol["rmu80_fingerprint"],
        "RMU80",
    )
    if set(base_fp) != set(rmu_fp):
        raise RuntimeError("原模型与RMU80参数集合不一致")

    rmu_cache = {name: parameters[name].detach().clone() for name in update_names}
    for row in update_rows:
        if base_fp[row["name"]] != row["base_tensor_sha256"]:
            raise RuntimeError(f"增量记录的基模参数指纹不符：{row['name']}")
        if rmu_fp[row["name"]] != row["trained_tensor_sha256"]:
            raise RuntimeError(f"增量记录的RMU80参数指纹不符：{row['name']}")

    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("模型没有处于冻结的eval状态")
    model.generation_config.do_sample = False
    for sampling_field in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, sampling_field):
            setattr(model.generation_config, sampling_field, None)

    context_limit = min(
        int(model.config.max_position_embeddings), int(tokenizer.model_max_length)
    )
    state = {"active": "rmu80"}
    return tokenizer, model, context_limit, update_rows, parameters, base_cache, rmu_cache, state, tensor_hash


def activate_state(target: str, update_rows, parameters, base_cache, rmu_cache, state, tensor_hash) -> None:
    if target == state["active"]:
        return
    expected_field = "base_tensor_sha256" if target == "original" else "trained_tensor_sha256"
    cache = base_cache if target == "original" else rmu_cache
    print(f"切换到{'原模型' if target == 'original' else 'RMU80'}并校验更新参数…", flush=True)
    with torch.no_grad():
        for name, cached_parameter in cache.items():
            parameters[name].copy_(cached_parameter)
    for row in update_rows:
        name = row["name"]
        if tensor_hash(parameters[name]) != row[expected_field]:
            raise RuntimeError(f"切换后参数指纹不匹配，拒绝继续：{name}")
    state["active"] = target


def eos_ids(model) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    return {int(eos)} if isinstance(eos, int) else {int(token) for token in eos}


def read_question() -> str | None:
    try:
        question = input("\n你：")
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。", flush=True)
        return None
    if question.strip() != "/multi":
        return question

    print("多行输入：粘贴问题，单独输入 /send 提交，/cancel 取消。", flush=True)
    lines = []
    try:
        while True:
            line = input("... ")
            if line.strip() == "/send":
                return "\n".join(lines)
            if line.strip() == "/cancel":
                print("已取消多行输入。", flush=True)
                return ""
            if line.strip() == "/exit":
                print("已退出。", flush=True)
                return None
            lines.append(line)
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。", flush=True)
        return None


def generate_one(model, tokenizer, inputs, input_tokens: int, max_new_tokens: int, end_ids: set[int], pad_id):
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            eos_token_id=sorted(end_ids) if end_ids else None,
            pad_token_id=pad_id,
        )
    generated_ids = output[0, input_tokens:].tolist()
    ended_naturally = bool(generated_ids and generated_ids[-1] in end_ids)
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    hit_limit = len(generated_ids) >= max_new_tokens and not ended_naturally
    return answer, hit_limit


def chat(tokenizer, model, context_limit, max_new_tokens, update_rows, parameters, base_cache, rmu_cache, state, tensor_hash):
    end_ids = eos_ids(model)
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id
    print(
        "双模型对比已就绪。每个问题独立评估，不保留对话历史；/multi 多行输入，/exit 退出。",
        flush=True,
    )

    while True:
        question = read_question()
        if question is None:
            return
        if not question.strip() or question.strip() == "/cancel":
            continue
        if question.strip() == "/exit":
            print("已退出。", flush=True)
            return

        try:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            )
        except Exception as exc:
            print(f"输入处理失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue

        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + max_new_tokens > context_limit:
            print(
                f"问题过长（输入 {input_tokens} + 输出预留 {max_new_tokens} > 上限 {context_limit}），未截断。",
                flush=True,
            )
            continue
        inputs = inputs.to(model.device)

        print("【原模型】", flush=True)
        try:
            activate_state("original", update_rows, parameters, base_cache, rmu_cache, state, tensor_hash)
            print("原模型正在生成回答，请稍候…", flush=True)
            original_answer, original_limited = generate_one(
                model, tokenizer, inputs, input_tokens, max_new_tokens, end_ids, pad_id
            )
            print(original_answer, flush=True)
            if original_limited:
                print("（原模型回答达到长度上限）", flush=True)
        except KeyboardInterrupt:
            print("\n已退出。", flush=True)
            return
        except Exception as exc:
            print(f"原模型生成失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

        print("【RMU80】", flush=True)
        try:
            activate_state("rmu80", update_rows, parameters, base_cache, rmu_cache, state, tensor_hash)
            print("RMU80正在生成回答，请稍候…", flush=True)
            rmu_answer, rmu_limited = generate_one(
                model, tokenizer, inputs, input_tokens, max_new_tokens, end_ids, pad_id
            )
            print(rmu_answer, flush=True)
            if rmu_limited:
                print("（RMU80回答达到长度上限）", flush=True)
        except KeyboardInterrupt:
            print("\n已退出。", flush=True)
            return
        except Exception as exc:
            print(f"RMU80生成失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="逐题比较原模型与RMU80回答")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="每个模型每题的最大新生成token数（默认：128）",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须是正整数")

    try:
        (
            tokenizer,
            model,
            context_limit,
            update_rows,
            parameters,
            base_cache,
            rmu_cache,
            state,
            tensor_hash,
        ) = load_compare_model()
        if args.max_new_tokens >= context_limit:
            raise ValueError(
                f"max-new-tokens={args.max_new_tokens} 必须小于上下文上限 {context_limit}"
            )
    except Exception as exc:
        print(
            f"双模型加载或完整指纹校验失败，已停止：{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        f"已验证原模型和RMU80；只缓存并切换 {len(update_rows)} 个更新参数，"
        f"单一模型实例位于 {model.device}。",
        flush=True,
    )
    chat(
        tokenizer,
        model,
        context_limit,
        args.max_new_tokens,
        update_rows,
        parameters,
        base_cache,
        rmu_cache,
        state,
        tensor_hash,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
