# Formal RMU plan v1 — prepared, not started

Prepared on 2026-10-13 (Asia/Shanghai). The real three-step CUDA trial passed; no formal model has been trained yet. This file fixes the first formal run before observing post-training official test results.

## Data and sampling

Use the previously cleaned, group-separated Bio splits. `scripts/prepare_rmu_formal_blocks.py` selects separately per role/split by the smallest SHA256(formal-v1:42:sample_id). Training candidate pools have2,048 articles per role; validation has64 per role. One native-tokenizer block is selected per article by a fixed hash over the sample ID, from full non-overlapping256-token blocks. Longer-article incomplete tails are ignored; shorter articles are retained if at least2 tokens. No cross-article packing or decode/re-encode occurs; no chat/system template is added. The first token label is ignored, consistent with the trial pretraining-style masks.

| Role | Split | Blocks | Blocks shorter than256 |
| --- | --- | ---: | ---: |
| Forget | train | 2048 | 4 |
| Retain | train | 2048 | 127 |
| Forget | validation | 64 | 0 |
| Retain | validation | 64 | 5 |

This is article-balanced subset training, not full-corpus coverage or uniform sampling of all corpus tokens. Counts are candidate-pool sizes, not actual usage. The forget dataset anchors2048 pairs; HF Trainer shuffles it, while OpenUnlearning randomly samples retain blocks with replacement.80 steps with16 microbatches consumes1280 pairs; the unique retain count must be measured, not presumed. Every microbatch index is recorded and joined to local block/source IDs.

Restricted token files stay under `data/processed/bio_v1/formal_blocks_v1`. Public `logs/rmu_formal_blocks_manifest.json` contains source/output checksums, tokenization identity and aggregate counts. Train/validation sample and article-group overlap are0. Source corpora and processed text remain intact.

## Frozen training configuration

Config: `open-unlearning/configs/experiment/unlearn/wmdp/qwen3_4b_bio_formal.yaml`; resolved snapshot: `logs/rmu_formal_resolved_config.json`.

- Original official Qwen3-4B-Instruct-2507; never continue trial weights. Local exact model contents are pinned in `logs/base_model_manifest.json`; the initial download did not record an immutable remote revision, which remains null. Before training, actual parameter fingerprints must equal the already-verified original baseline; token files/tokenizer hashes are also checked.
- BF16, SDPA, cache disabled, single4090D, batch1, sequence cap256, gradient accumulation16 (effective batch16 forget/retain pairs).
- AdamW, lr5e-5, constant scheduler, no warmup, weight_decay0, max_grad_norm1.0, seed/data_seed42, max_steps80.
- RMU layer7 hook; only layers5/6/7 MLP down_proj.weight; gamma1, alpha1, steering_coeff2, retain loss EMBED_DIFF; frozen reference deepcopy on accelerator device.
- No DeepSpeed, LoRA, quantization or gradient checkpointing. The smoke memory result applies to this sequence/batch setting; accumulation still requires measurement during the formal run.
- No automatic WMDP/MMLU evaluator during training. Save only the final step80 full model in <=1GB shards, with tokenizer/configs and fingerprints; no intermediate full checkpoints or early stopping.

RMU now explicitly declares that it does not use token-count loss kwargs, so Trainer divides microbatch losses by the actual accumulation count. A tiny CPU check verified16 accumulated gradients equal a single duplicate-batch gradient. The forward-cache helper respects an outer no_grad context during validation.

## Independent validation and telemetry

`scripts/rmu_validation.py` monitors the fixed64+64 held-out blocks at steps0/20/40/60/80. It records forget/retain representation losses, weighted total representation loss, and token-weighted retain NLL/PPL from the current model. These are Bio held-out metrics, not MMLU or a safety score. The same training control vector is used across validation checkpoints. Validation runs without gradients, restores training mode, and cannot change parameters or the accumulated training metrics.

The fixed checkpoint is step80 regardless of validation scores; validation detects failures/drift, not a search over WMDP/MMLU. Non-finite losses abort; retain PPL is null if exponentiation would overflow. CUDA/CPU inapplicable fields are null. Training logs include loss components, LR, grad norm, tokens, actual block indices, throughput, timestamps, elapsed time and cumulative CUDA allocated/reserved peaks. Validation rows and training rows have distinct phase fields.

## Checks and execution

- `tests/check_rmu_safety.py`: selected gradients/reference freezing, numeric masked loss, empty masks, hook cleanup and outer no_grad.
- `tests/check_rmu_formal_monitor.py`: cached token preservation, sample indices,16-microbatch scaling, validation with unchanged parameters/no gradients, token-weighted NLL and mode restoration. Passed on tiny CPU Qwen3; this is not formal CUDA training.
- `scripts/check_rmu_formal_readiness.py`: actual architecture via meta tensors, composed TrainingArguments, all cache SHA/token ranges/labels, representative source re-tokenization and collation. Passed; no4B weights loaded.
- Logs include unsuccessful preparation assertions and the stopped slow check. Explicit null clears inline inherited test evaluators. Vocabulary size is cached once to avoid expensive per-token queries.

Execution script is retained at `scripts/run_rmu_formal.py`; when authorized for the next step, run it from the project root with the independent environment. It refuses to overwrite the run directory, checks the initial baseline fingerprints, records actual consumed indices and validation rows, saves one final model and checks parameter-only changes and exact reloading/finite forward. It does not run post-training official tests.

After formal training: review logs/validation and saved checkpoint, then evaluate full WMDP-Bio and MMLU under their locked baseline protocols. Formal full-weight GitHub LFS/hosting and authorized data-delivery access still need setup; source restrictions are not waived by storage configuration.
