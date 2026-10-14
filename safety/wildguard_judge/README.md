# WildGuard automatic judge

This package scores the already-generated 512-token four-group replies with `allenai/wildguard` as an offline automatic evaluator. It does not alter the Qwen3Guard protection flow. The pinned revision, official template/parser sources, generation configuration, local-only paths, and blocked-access status are recorded in `config.json` and `REPORT.md`.

The Hugging Face repository is gated. The local account could read repository metadata but the pinned file request returned HTTP 403 (“not in the authorized list”). The official access page is <https://huggingface.co/allenai/wildguard>. Complete the AI2 Responsible Use Guidelines/access form there with the same Hugging Face account used by this environment before downloading. No model weights were downloaded. Do not swap in another model to bypass this gate.

After file access is granted, run from the project root:

```bash
envs/unlearning/bin/python -m safety.wildguard_judge.download
envs/unlearning/bin/python -m safety.wildguard_judge.prepare
envs/unlearning/bin/python -m safety.wildguard_judge.run --phase pilot
```

Inspect `pilot_check.json`; only if all eight real WildGuardTest pairs finish with EOS and all three official fields parse, continue:

```bash
envs/unlearning/bin/python -m safety.wildguard_judge.run --phase full
envs/unlearning/bin/python -m safety.wildguard_judge.summarize
envs/unlearning/bin/python -m safety.wildguard_judge.verify
```

Inference caches, original text, answers, per-item scores, mappings, and the blind review form stay under the ignored `cache/wildguard_judge_v1/`. Only model metadata, fixed protocol code, aggregate scores, hashes, and reports belong in Git. `run.py` commits each result before moving to the next pair; rerunning `--phase full` reuses its cache.

The checkpoint was not run here because model files were denied by Hugging Face. The preflight script verifies local answer mappings, fixed refusals, label manifests, inference-input isolation, and the official three-field parser fixture without contacting the judge.
