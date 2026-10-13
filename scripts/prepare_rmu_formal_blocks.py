"""Build restricted, deterministic per-article token blocks for the first formal RMU run."""
import json,hashlib,heapq
from pathlib import Path
from datetime import datetime,timezone,timedelta
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/processed/bio_v1';OUT=BASE/'formal_blocks_v1';SEED=42;LENGTH=256
if OUT.exists():raise RuntimeError('Refuse to overwrite prepared formal blocks.')
OUT.mkdir()
tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507',local_files_only=True)
report={'prepared_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'version':'rmu-formal-blocks-v1',
        'seed':SEED,'sequence_length':LENGTH,'source_processing_version':'bio-clean-v1','training_started':False,
        'selection_rule':'Smallest SHA256(formal-v1:42:sample_id), separately per role and split; one hash-selected full 256-token block per article; shorter articles kept if >=2 tokens; incomplete tails ignored for longer articles.',
        'tokenization':'Native tokenizer add_special_tokens=True; no chat template, no decode/re-encode, no cross-article packing',
        'files':[],'tokenizer_files':[]}
for name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']:
 p=ROOT/'models/Qwen3-4B-Instruct-2507'/name
 report['tokenizer_files'].append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
for role in ['forget','retain']:
 for split,count in [('train',2048),('validation',64)]:
  source=BASE/role/f'{split}.jsonl';heap=[]
  for line in source.open():
   row=json.loads(line);rank=int(hashlib.sha256(f"formal-v1:{SEED}:{row['sample_id']}".encode()).hexdigest(),16)
   entry=(-rank,row['sample_id'],row)
   if len(heap)<count:heapq.heappush(heap,entry)
   elif rank < -heap[0][0]:heapq.heapreplace(heap,entry)
  selected=sorted(heap,key=lambda x:(-x[0],x[1]));dest=OUT/role/f'{split}.jsonl';dest.parent.mkdir(exist_ok=True)
  actual=0;short=0;skipped=[];source_tokens=0;block_tokens=0
  with dest.open('w') as f:
   for _,sid,row in selected:
    ids=tokenizer(row['text'],add_special_tokens=True)['input_ids'];source_tokens+=len(ids)
    if len(ids)<2:skipped.append(sid);continue
    full=len(ids)//LENGTH
    chosen=int(hashlib.sha256(f'block:{SEED}:{sid}'.encode()).hexdigest(),16)%full if full else 0
    start=chosen*LENGTH;block=ids[start:start+LENGTH];short+=len(block)<LENGTH;block_tokens+=len(block)
    result={'block_id':f'{sid}-tok-{start}-{start+len(block)}','sample_id':sid,'article_group_id':row['article_group_id'],
            'source_file':row['source_file'],'source_row_index':row['source_row_index'],'role':role,'split':split,
            'token_start':start,'token_end':start+len(block),'source_token_count':len(ids),'input_ids':block,
            'used_for_training':False}
    f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');actual+=1
  report['files'].append({'path':str(dest.relative_to(ROOT)),'source_path':str(source.relative_to(ROOT)),
                         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                         'role':role,'split':split,'selected_articles':len(selected),'blocks':actual,
                         'short_blocks':short,'skipped_lt2tokens':len(skipped),'selected_article_tokens':source_tokens,
                         'block_tokens':block_tokens,'size_bytes':dest.stat().st_size,
                         'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
  print(role,split,'blocks',actual,'short',short,flush=True)
  if split=='train':assert actual>=1280
  else:assert actual>0
sets={};groups={}
for info in report['files']:
 key=(info['role'],info['split']);rows=[json.loads(l) for l in (ROOT/info['path']).open()]
 sets[key]={r['sample_id'] for r in rows};groups[key]={r['article_group_id'] for r in rows}
 for r in rows:assert 2<=len(r['input_ids'])<=LENGTH
train=set.union(*(sets[r,'train'] for r in ['forget','retain']));val=set.union(*(sets[r,'validation'] for r in ['forget','retain']))
assert not train & val
train_groups=set.union(*(groups[r,'train'] for r in ['forget','retain']));val_groups=set.union(*(groups[r,'validation'] for r in ['forget','retain']))
assert not train_groups & val_groups
report['checks']={'train_validation_sample_overlap':0,'train_validation_article_group_overlap':0,'valid_token_lengths':True}
report['limitations']=['Article-balanced subset and one block per article, not full-corpus training.',
                      'Candidate pool size is not actual number of sampled/consumed training articles; actual indices must be recorded during training.',
                      'Validation is from held-out Bio corpora; representation loss is not MMLU or proof of safety.']
(ROOT/'logs/rmu_formal_blocks_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print('Prepared and verified; no training started.',flush=True)
