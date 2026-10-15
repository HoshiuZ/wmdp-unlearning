#!/usr/bin/env python3
"""Interactive local chat with the pinned RMU80 checkpoint."""

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


def verify_fingerprints(
    model, tensor_hash, expected_path: Path, expected_digest: str, label: str
) -> int:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    named_parameters = list(model.named_parameters())
    actual = {}
    for index, (name, parameter) in enumerate(named_parameters, start=1):
        actual[name] = tensor_hash(parameter)
        if index % 40 == 0 or index == len(named_parameters):
            print(
                f"{label}参数指纹校验：{index}/{len(named_parameters)}",
                flush=True,
            )
    if actual != expected:
        mismatches = sorted(
            name
            for name in expected.keys() | actual.keys()
            if expected.get(name) != actual.get(name)
        )
        preview = ", ".join(mismatches[:3]) or "unknown mismatch"
        raise RuntimeError(f"参数指纹与 {expected_path.relative_to(ROOT)} 不一致：{preview}")
    actual_digest = digest_json(actual)
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"参数指纹摘要不匹配：期望 {expected_digest}，实际 {actual_digest}"
        )
    return len(actual)


def load_rmu80():
    if not BASE.is_dir():
        raise FileNotFoundError(f"找不到基模目录：{BASE}")
    if not DELTA.is_dir():
        raise FileNotFoundError(f"找不到RMU80增量目录：{DELTA}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用CUDA设备；本脚本要求使用本地GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持BF16，已停止加载")

    protocol = json.loads((FOUR_GROUPS / "manifest.json").read_text(encoding="utf-8"))
    base_fingerprints = LOCAL / "base_fingerprints.json"
    rmu_fingerprints = LOCAL / "rmu80_fingerprints.json"
    if not base_fingerprints.is_file() or not rmu_fingerprints.is_file():
        raise FileNotFoundError("缺少四组实验保存的基模或RMU80参数指纹文件")

    print("正在读取本地tokenizer…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tokenizer.padding_side = "left"
    template = tokenizer.chat_template
    template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
    if template_sha != protocol["chat_template_sha256"]:
        raise RuntimeError("本地tokenizer聊天模板与四组实验记录的模板指纹不一致")

    print("正在从本地加载基模权重（BF16，SDPA）…", flush=True)
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

    print("基模权重已加载，开始逐项校验基模指纹…", flush=True)
    base_count = verify_fingerprints(
        model,
        tensor_hash,
        base_fingerprints,
        protocol["base_fingerprint"],
        "基模",
    )
    print("基模指纹通过，正在应用RMU80增量…", flush=True)
    load_delta(model, DELTA)
    print("增量已应用，开始逐项校验RMU80指纹…", flush=True)
    rmu_count = verify_fingerprints(
        model,
        tensor_hash,
        rmu_fingerprints,
        protocol["rmu80_fingerprint"],
        "RMU80",
    )
    if base_count != rmu_count:
        raise RuntimeError("基模与RMU80指纹参数数量不一致")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("RMU80模型未处于冻结的eval状态")
    model.generation_config.do_sample = False
    for sampling_field in ("temperature", "top_p", "top_k"):
        if hasattr(model.generation_config, sampling_field):
            setattr(model.generation_config, sampling_field, None)

    context_limit = min(
        int(model.config.max_position_embeddings), int(tokenizer.model_max_length)
    )
    return tokenizer, model, context_limit, rmu_count


def eos_ids(model) -> set[int]:
    eos = model.generation_config.eos_token_id
    if eos is None:
        return set()
    return {int(eos)} if isinstance(eos, int) else {int(token) for token in eos}


def chat(tokenizer, model, context_limit: int, max_new_tokens: int) -> None:
    history: list[dict[str, str]] = []
    end_ids = eos_ids(model)
    pad_id = model.generation_config.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    print("RMU80已加载。输入 /clear 清空上下文，输入 /exit 退出。", flush=True)
    while True:
        try:
            question = input("你：")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。", flush=True)
            return

        if not question.strip():
            continue
        command = question.strip()
        if command == "/exit":
            print("已退出。", flush=True)
            return
        if command == "/clear":
            history.clear()
            print("已清空聊天上下文。", flush=True)
            continue

        messages = history + [{"role": "user", "content": question}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
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
                f"上下文过长（输入 {input_tokens} + 输出预留 {max_new_tokens} > 上限 {context_limit}）。"
                "请先输入 /clear 清空聊天上下文。",
                flush=True,
            )
            continue

        inputs = inputs.to(model.device)
        try:
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
        except KeyboardInterrupt:
            print("\n已退出。", flush=True)
            return
        except Exception as exc:
            print(f"生成失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            continue

        print(f"模型：{answer}", flush=True)
        if len(generated_ids) >= max_new_tokens and not ended_naturally:
            print("（已达到输出长度上限）", flush=True)
        history.extend(
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="在终端与本地RMU80模型聊天")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="单轮最大新生成token数（默认：1024）",
    )
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens 必须是正整数")

    try:
        tokenizer, model, context_limit, parameter_count = load_rmu80()
        if args.max_new_tokens >= context_limit:
            raise ValueError(
                f"max-new-tokens={args.max_new_tokens} 必须小于模型上下文上限 {context_limit}"
            )
    except Exception as exc:
        print(
            f"RMU80加载失败，已停止且未回退到原模型：{type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    print(
        f"已校验并加载RMU80（{parameter_count}个参数，BF16，SDPA，{model.device}）。",
        flush=True,
    )
    chat(tokenizer, model, context_limit, args.max_new_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
