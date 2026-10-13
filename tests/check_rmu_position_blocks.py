import sys,json,random
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from rmu_position_blocks import PositionBlockDataset
from rmu_adaptation_utils import dump
from transformers import AutoTokenizer
base=ROOT/'data/processed/bio_v1';tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507',local_files_only=True)
seq=json.loads((ROOT/'open-unlearning/saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/consumed_block_indices.json').read_text());result={}
for role,indices in seq.items():
 old=[json.loads(l) for l in (base/f'formal_blocks_v1/{role}/train.jsonl').open()];ds=PositionBlockDataset(base/f'position_blocks_v2/{role}/train.jsonl',role,tok)
 assert [r['sample_id'] for r in ds.rows]==[r['sample_id'] for r in old]
 before=torch.get_rng_state().clone();py=random.getstate();tokens=0
 for index in indices:
  row=ds[index];assert len(row['input_ids'])==len(old[index]['input_ids']);assert row['labels'][0]==-100 and torch.equal(row['labels'][1:],row['input_ids'][1:]);tokens+=len(row['input_ids'])
 assert torch.equal(before,torch.get_rng_state()) and py==random.getstate()
 assert tokens==sum(len(old[i]['input_ids']) for i in indices)
 unique={(r['sample_id'],r['token_start'],r['token_end']) for r in ds.exposures};assert len(unique)>=len(set(indices))
 assert any(not r['original_position'] for r in ds.exposures)
 result[role]={'matched_input_tokens':tokens,'matched_article_sequence':True,'unchanged_rng':True,'actual_unique_positions':len(unique),'old_unique_positions':len(set(indices)),'non_original_exposures':sum(not r['original_position'] for r in ds.exposures)}
dump(ROOT/'logs/rmu_scale_v2_coverage_readiness.json',result);print(result)
