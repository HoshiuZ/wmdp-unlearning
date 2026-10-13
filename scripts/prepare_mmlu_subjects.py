"""Recreate subject Parquet files from the pinned, archived MMLU all splits.

Run with envs/unlearning/bin/python scripts/prepare_mmlu_subjects.py.
Sources and revision are recorded in logs/mmlu_dataset_manifest.json.
No question, answer or ordering within each subject is modified.
"""
from pathlib import Path
import hashlib
import json
import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "logs/mmlu_dataset_manifest.json").read_text())
for source in manifest["source_files"]:
    path = root / source["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path
for split in ("test", "dev"):
    table = pq.read_table(root / f"data/evaluation/mmlu/source/all/{split}-00000-of-00001.parquet")
    for subject in sorted(set(table["subject"].to_pylist())):
        subset = table.filter(pc.equal(table["subject"], subject))
        assert len(subset) == manifest["subject_counts"][split][subject]
        path = root / f"data/evaluation/mmlu/subjects/{subject}/{split}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(subset, path)
print("Recreated pinned MMLU subject snapshots.")
