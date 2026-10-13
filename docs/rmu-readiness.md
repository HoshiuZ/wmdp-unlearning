# RMU readiness review

Reviewed 2026-10-13 (Asia/Shanghai). No actual Qwen3-4B training has been executed.

## Implementation and fixes

RMU hooks `model.layers.7`, steers masked forget activations toward a fixed random vector and preserves retain activations against a reference copy. Total loss is gamma * forget representation MSE + alpha * retain representation MSE. These losses are not NLL/PPL. For the proposed trial gamma=1, alpha=1, steering_coeff=2; these are example starting values, not tuned Qwen optima.

The reference is a deepcopy placed on the accelerator device. Two full BF16 models are needed in the current implementation; freezing reduces gradients/optimizer state, not reference-weight storage. GPU peak memory and training compatibility remain unverified.

The optimizer previously restored requires_grad on all model parameters after selecting an optimizer subset. This was fixed to preserve the chosen gradient mask, explicitly freeze reference weights and fail on an empty parameter match. Also corrected batch>1 masked-loss denominator broadcasting and ensured hooks are removed on a failed forward. Batch1 loss formula is unchanged.

CPU regression checks in `tests/check_rmu_safety.py` use a tiny random Qwen3 model: exactly selected optimizer parameters, frozen reference, finite backward and selected gradient flow, no gradients on frozen parameters, batched masked-loss numeric result, all-masked guard and failed-forward hook cleanup all passed. This is not a test of real 4B CUDA training.

## Qwen and dataset verification

`scripts/check_rmu_readiness.py` uses meta tensors to inspect the actual local Qwen3 config without loading weights. Exactly three weights match: layers5/6/7 `mlp.down_proj.weight`; the single hook is layer7. Exact counts/shapes are in `logs/rmu_readiness_report.json`.

Full-corpus PretrainingDataset concatenates and tokenizes all text and decodes/re-encodes chunks. It should not be invoked blindly for the trial. The trial uses CompletionDataset, JSON loader and text field on eight articles per role. This takes the leading256 tokens per article and is only a compatibility trial; the formal token-block sampling strategy has not been fixed. It adds no chat/system template.

`scripts/prepare_rmu_smoke_subset.py` chooses eight training records per role by smallest SHA256(42:sample_id), preserving IDs and text locally under `data/processed/bio_v1/smoke_input`. No validation/evaluation record is used. Selection/checksums are in `logs/rmu_smoke_subset_manifest.json`. Config composition, TrainingArguments, correct source text, all sample masks and nested collator shapes [1,256] were verified. An initial Hydra include-path failure is archived separately.

## Prepared trial and next step

Config: `open-unlearning/configs/experiment/unlearn/wmdp/qwen3_4b_bio_smoke.yaml`; resolved config: `logs/rmu_smoke_resolved_config.json`. Proposed run: batch1, gradient accumulation1, sequence256, three optimizer steps, lr5e-5, AdamW, BF16/SDPA, seed42, no gradient checkpointing or DeepSpeed, no automatic MMLU/WMDP evaluation, no intermediate checkpoints, final save by train.py. remove_unused_columns=false preserves nested forget/retain inputs.

Before the trial runs, add actual per-step forget/retain loss and memory telemetry; then verify finite forward/backward, peak CUDA memory, selected-weight changes, final save and reload. If it fails, preserve logs and resolve the concrete error. Do not evaluate official tests to select trial hyperparameters. A later formal run must independently restart from the original official model and cannot reuse trial weights as an equivalent baseline.
