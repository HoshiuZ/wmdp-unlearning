"""Select eight training articles per role deterministically; restricted text stays local."""
import hashlib
import heapq
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/bio_v1'
OUT=BASE/'smoke_input'
if OUT.exists(): raise RuntimeError('Smoke inputs already exist; do not overwrite.')
OUT.mkdir()
manifest={'selection':'Eight smallest SHA256(42:sample_id) per role; training split only',
          'seed':42,'count_per_role':8,'max_sequence_length':256,'training_started':False,'roles':{}}
for role in ('forget','retain'):
    source=BASE/role/'train.jsonl'; selected=[]
    with source.open() as f:
        for line in f:
            row=json.loads(line)
            rank=int(hashlib.sha256(('42:'+row['sample_id']).encode()).hexdigest(),16)
            entry=(-rank,row['sample_id'],line)
            if len(selected)<8:heapq.heappush(selected,entry)
            elif rank < -selected[0][0]:heapq.heapreplace(selected,entry)
    selected.sort(key=lambda x:(-x[0],x[1]))
    dest=OUT/f'{role}.jsonl'
    dest.write_text(''.join(x[2] for x in selected))
    manifest['roles'][role]={'source':str(source.relative_to(ROOT)),'path':str(dest.relative_to(ROOT)),
                              'sample_ids':[x[1] for x in selected],'size_bytes':dest.stat().st_size,
                              'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
(ROOT/'logs/rmu_smoke_subset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Prepared eight train articles per role; no validation or test samples selected.')
