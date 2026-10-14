"""Download only the pinned WildGuard BF16 safetensors and tokenizer/config once."""
import os
import json
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download

from .common import CFG, HERE, MODEL_SNAPSHOT, ROOT, dump


ALLOW = [
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
    "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
]


def main():
    token = get_token()
    if not token:
        raise RuntimeError("No local Hugging Face token is available for this gated repository")
    info = HfApi(token=token).model_info(CFG["model_id"], revision=CFG["revision"], files_metadata=True)
    if info.sha != CFG["revision"] or info.gated not in ("auto", True):
        raise RuntimeError("Pinned revision/access state differs from the reviewed model metadata")
    wanted = {s.rfilename: s for s in info.siblings if s.rfilename in ALLOW}
    if set(wanted) != set(ALLOW):
        raise RuntimeError("Official revision does not contain the required inference files")
    expected_weight_bytes = sum(int(wanted[name].size) for name in ALLOW if name.endswith(".safetensors"))
    print(json.dumps({"download_start": True, "revision": info.sha,
                      "safetensors_bytes": expected_weight_bytes,
                      "requested_files": len(ALLOW), "excluded_duplicate_bin_weights": True}, flush=True))
    cache_dir = str(ROOT / "cache/huggingface/hub")
    try:
        # Probe a tiny config file first so a gated 403 cannot leave large partial weight files.
        hf_hub_download(repo_id=CFG["model_id"], filename="config.json", revision=CFG["revision"],
                        cache_dir=cache_dir, token=token)
        snapshot_download(repo_id=CFG["model_id"], revision=CFG["revision"],
                          cache_dir=cache_dir, allow_patterns=ALLOW,
                          token=token, max_workers=2)
    except Exception as exc:
        from huggingface_hub.errors import GatedRepoError
        if isinstance(exc, GatedRepoError):
            raise RuntimeError("WildGuard is gated and this account is not authorized to download files. "
                               "Accept the AI2 Responsible Use Guidelines at https://huggingface.co/allenai/wildguard "
                               "then rerun this command.") from None
        raise
    if not MODEL_SNAPSHOT.is_dir():
        raise RuntimeError("Pinned snapshot did not appear in the configured shared cache")
    files = {}
    for name in ALLOW:
        path = MODEL_SNAPSHOT / name
        if not path.is_file():
            raise RuntimeError(f"Required model file is absent: {name}")
        files[name] = {"bytes": path.stat().st_size, "is_symlink": path.is_symlink()}
        remote_size = wanted[name].size
        if remote_size is not None and path.stat().st_size != remote_size:
            raise RuntimeError(f"Cached model file size mismatch: {name}")
    weight_bytes = sum(value["bytes"] for name, value in files.items() if name.endswith(".safetensors"))
    receipt = {"model_id": CFG["model_id"], "revision": info.sha,
               "license": CFG["license"], "gated_access_verified": True,
               "downloaded_cache_path": str(MODEL_SNAPSHOT.relative_to(ROOT)),
               "files": files, "safetensors_bytes": weight_bytes,
               "total_selected_bytes": sum(value["bytes"] for value in files.values()),
               "duplicate_pytorch_bin_weights_downloaded": False,
               "repository_gated": bool(info.gated)}
    dump(HERE / "model_source.json", receipt)
    dump(ROOT / CFG["local_root"] / "download_receipt.json", receipt)
    print(json.dumps({"snapshot_ready": True, "safetensors_bytes": weight_bytes,
                      "total_selected_bytes": receipt["total_selected_bytes"],
                      "snapshot": receipt["downloaded_cache_path"]}, flush=True))


if __name__ == "__main__":
    main()
