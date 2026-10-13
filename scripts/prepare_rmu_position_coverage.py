"""Prepare alternative blocks of same training articles; same per-article lengths."""
import json,hashlib,shutil
from pathlib import Path
from transformers import AutoTokenizer
from rmu_adaptation_utils import ROOT,BASE,dump
OUT=ROOT/'data/processed/bio_v1/position_blocks_v2'
def main():
 assert not OUT.exists(),'Do not overwrite data version';assert shutil.disk_usage(ROOT).free>2**30+100*2**20;OUT.mkdir()
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);manifest={'version':'rmu-position-v2','seed':42,'max_options':4,'base_data_version':'rmu-formal-blocks-v1','selection':'Same ordered 2048 original training articles; original block plus up to3 hash-ranked distinct nonoverlapping complete256-token blocks. Short lengths unchanged. First visit uses non-original when possible; repeats deterministic cyclic positions; no global RNG consumption.','no_evaluation_documents_added':True,'files':[]}
 validation_groups=set()
 for role in ['forget','retain']:
  for line in (ROOT/f'data/processed/bio_v1/{role}/validation.jsonl').open():validation_groups.add(json.loads(line)['article_group_id'])
 for role in ['forget','retain']:
  old=ROOT/f'data/processed/bio_v1/formal_blocks_v1/{role}/train.jsonl';rows=[json.loads(l) for l in old.open()];needed={r['sample_id'] for r in rows};texts={}
  source=ROOT/f'data/processed/bio_v1/{role}/train.jsonl'
  for line in source.open():
   r=json.loads(line)
   if r['sample_id'] in needed:texts[r['sample_id']]=r['text']
  assert set(texts)==needed;dest=OUT/role/'train.jsonl';dest.parent.mkdir();multi=0;totaloptions=0
  with dest.open('w') as f:
   for row in rows:
    assert row['article_group_id'] not in validation_groups
    ids=tok(texts[row['sample_id']],add_special_tokens=True)['input_ids'];assert ids[row['token_start']:row['token_end']]==row['input_ids']
    options=[{'token_start':row['token_start'],'token_end':row['token_end'],'input_ids':row['input_ids']}]
    if len(row['input_ids'])==256:
     candidates=[i*256 for i in range(len(ids)//256) if i*256!=row['token_start']]
     candidates.sort(key=lambda i:hashlib.sha256(f'coverage-v2:42:{row["sample_id"]}:{i}'.encode()).hexdigest())
     for start in candidates[:3]:options.append({'token_start':start,'token_end':start+256,'input_ids':ids[start:start+256]})
    result={**row,'options':options};f.write(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n');multi+=len(options)>1;totaloptions+=len(options)
  manifest['files'].append({'role':role,'path':str(dest.relative_to(ROOT)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,'articles':len(rows),'multiple_position_articles':multi,'total_options':totaloptions,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'original_cache_sha256':hashlib.sha256(old.read_bytes()).hexdigest()})
  print(role,multi,totaloptions,flush=True)
 manifest['checks']={'same_article_order':True,'same_per_article_token_length':True,'train_validation_group_overlap':0,'original_tokenization_verified':True}
 dump(OUT/'manifest.json',manifest);dump(ROOT/'logs/rmu_scale_v2_coverage_manifest.json',manifest)
if __name__=='__main__':main()
