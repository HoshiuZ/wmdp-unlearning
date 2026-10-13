"""Deterministic per-visit block positions without consuming sampler RNG."""
import json,hashlib
from pathlib import Path
import torch
class PositionBlockDataset:
 def __init__(self,file,role,tokenizer):
  self.rows=[json.loads(l) for l in Path(file).open()];self.role=role;self.visits={};self.exposures=[];vocab=len(tokenizer)
  assert len(self.rows)==2048
  for r in self.rows:
   assert all(len(o['input_ids'])==len(r['input_ids']) for o in r['options'])
   assert all(0<=v<vocab for o in r['options'] for v in o['input_ids'])
 def __len__(self):return len(self.rows)
 def __getitem__(self,index):
  row=self.rows[index];visit=self.visits.get(index,0);self.visits[index]=visit+1;n=len(row['options'])
  # First exposure selects a different position when possible; repeats rotate.
  offset=1+int(hashlib.sha256(f'coverage-v2:42:{self.role}:{row["sample_id"]}'.encode()).hexdigest(),16)%(n-1) if n>1 else 0
  chosen=(offset+visit)%n;o=row['options'][chosen];ids=torch.tensor(o['input_ids']);labels=ids.clone();labels[0]=-100
  self.exposures.append({'index':index,'visit':visit,'block_id':f'{row["sample_id"]}-tok-{o["token_start"]}-{o["token_end"]}','sample_id':row['sample_id'],'article_group_id':row['article_group_id'],'option':chosen,'token_start':o['token_start'],'token_end':o['token_end'],'input_tokens':len(ids),'original_position':chosen==0})
  return {'input_ids':ids,'labels':labels,'attention_mask':torch.ones_like(ids),'index':index}
