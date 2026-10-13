"""Recreate restricted actually-used block records from the formal consumption audit."""
from pathlib import Path
import json,collections,hashlib
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_formal80_v1'
BASE=ROOT/'data/processed/bio_v1';OUT=BASE/'formal_usage_v1'
consumed=json.loads((RUN/'consumed_block_indices.json').read_text())
manifest=json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())
OUT.mkdir(exist_ok=True)
for role in ('forget','retain'):
    info=next(x for x in manifest['files'] if x['role']==role and x['split']=='train')
    source=ROOT/info['path']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==info['sha256']
    rows=[json.loads(line) for line in source.open()]
    counts=collections.Counter(consumed[role]);assert sum(counts.values())==1280
    with (OUT/f'{role}_used_train_blocks.jsonl').open('w') as f:
        for index in sorted(counts):
            row=dict(rows[index]);row.update(used_for_training=True,exposure_count=counts[index],
                                            candidate_block_index=index,labels=[-100]+row['input_ids'][1:])
            f.write(json.dumps(row,allow_nan=False)+'\n')
    print(role,'unique_blocks',len(counts),'exposures',sum(counts.values()))
