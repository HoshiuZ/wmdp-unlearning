"""Helpers confined to the fixed 40/80/120 pure-RMU comparison."""
import contextlib,hashlib,json,math,shutil,time
from pathlib import Path
import torch
from rmu_adaptation_utils import ROOT,REPO,dump,tensor_hash,preserve_rng
HISTORY=REPO/'saves/unlearn/rmu_scale_v2/h12_a2_s32_fixed40'
RUN=REPO/'saves/unlearn/rmu_step_comparison_v1/h12_a2_s32_seed42_continuous120'
STEPS=(40,80,120)
SELECTED=[f'model.layers.{i}.mlp.down_proj.weight' for i in (10,11,12)]

def space(required=0):
 free=shutil.disk_usage(ROOT).free
 if free<2**30+required:raise RuntimeError(f'Insufficient disk reserve: free={free}, need={2**30+required}')
 return free

def state_hash(value):
 h=hashlib.sha256()
 def visit(x):
  if torch.is_tensor(x):h.update(str((x.dtype,tuple(x.shape))).encode());h.update(bytes.fromhex(tensor_hash(x.reshape(-1))))
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   for item in x:visit(item)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

@contextlib.contextmanager
def protected_callback(trainer,records,kind,step):
 """Preserve RNG and modes; assert weights, optimizer, scheduler and gradients unchanged."""
 start=time.perf_counter()
 models=[trainer.model,trainer.ref_model];modes=[(m,m.training) for model in models for m in model.modules()]
 params={n:tensor_hash(p) for n,p in trainer.model.named_parameters() if p.requires_grad}
 opt=state_hash(trainer.optimizer.state_dict()) if trainer.optimizer else None
 sched=state_hash(trainer.lr_scheduler.state_dict()) if trainer.lr_scheduler else None
 gradients={n:None if p.grad is None else tensor_hash(p.grad) for n,p in trainer.model.named_parameters()}
 last=trainer.last_loss_record;record_count=len(trainer.step_loss_records)
 try:
  with preserve_rng():yield
 finally:
  for m,training in modes:m.training=training
  trainer.last_loss_record=last
 assert params=={n:tensor_hash(p) for n,p in trainer.model.named_parameters() if p.requires_grad}
 assert opt==(state_hash(trainer.optimizer.state_dict()) if trainer.optimizer else None)
 assert sched==(state_hash(trainer.lr_scheduler.state_dict()) if trainer.lr_scheduler else None)
 assert gradients=={n:None if p.grad is None else tensor_hash(p.grad) for n,p in trainer.model.named_parameters()}
 assert len(trainer.step_loss_records)==record_count
 records.append({'kind':kind,'step':step,'seconds_including_integrity_checks':time.perf_counter()-start,'weights_optimizer_scheduler_gradients_unchanged':True,'rng_and_modes_restored':True})

class HistoricalDirection:
 def __init__(self,trainer,out):
  self.trainer=trainer;self.out=out;self.generate=trainer.get_control_vector
  stored=json.loads((HISTORY/'control_vector.json').read_text())
  self.vector=torch.tensor(stored['values'],dtype=torch.float32).reshape(1,1,-1)
  original=json.loads((REPO/'saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/control_vector.json').read_text())
  self.unit=torch.tensor(original['values'],dtype=torch.float32).reshape(1,1,-1)/2
  assert tensor_hash(self.vector)==stored['sha256'] and float(self.vector.norm())==32
  assert tensor_hash(self.unit)==stored['unit_direction_sha256'] and torch.equal(self.unit*32,self.vector)
 def __call__(self,dim):
  first=self.trainer.control_vec is None;before=torch.get_rng_state().clone()
  generated=self.generate(dim)
  if first:
   assert torch.allclose(generated/32,self.unit,atol=1e-7,rtol=0),'Control direction disagrees with historical RNG protocol'
   after=torch.get_rng_state().clone();assert not torch.equal(before,after)
   dump(self.out/'control_vector.json',{'values':self.vector.flatten().tolist(),'sha256':tensor_hash(self.vector),'l2_norm':float(self.vector.norm()),'unit_direction_sha256':tensor_hash(self.unit),'initial_cpu_rng_before_sha256':tensor_hash(before),'initial_cpu_rng_after_sha256':tensor_hash(after),'initialization':'First historical torch.rand draw consumed once at step0 validation, before generic RNG preservation. No early pre-seeding of control_vec.'})
  self.trainer.control_vec=self.vector.clone();return self.trainer.control_vec

def training_rows(path):return [json.loads(l) for l in path.open() if json.loads(l)['phase']=='train']
def exposure(rows,out,step):
 manifest=json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text());files={r['role']:ROOT/r['path'] for r in manifest['files'] if r['split']=='train'}
 result={'step':step,'input_tokens':sum(r['input_tokens'] for r in rows),'loss_tokens':sum(r['loss_tokens'] for r in rows),'roles':{}}
 for role in ('forget','retain'):
  blocks=[json.loads(l) for l in files[role].open()];indices=[i for r in rows for i in r[f'sampled_{role}_indices']]
  assert len(indices)==step*16
  counts={str(i):indices.count(i) for i in sorted(set(indices))}
  selected=[blocks[i] for i in indices]
  result['roles'][role]={'indices':indices,'exposures':len(indices),'unique_blocks':len(counts),'unique_articles':len({r['sample_id'] for r in selected}),'block_exposure_counts':counts,'input_tokens':sum(len(r['input_ids']) for r in selected),'effective_prediction_tokens':sum(len(r['input_ids'])-1 for r in selected)}
 assert result['input_tokens']==sum(r['input_tokens'] for r in result['roles'].values())
 assert result['loss_tokens']==sum(r['effective_prediction_tokens'] for r in result['roles'].values())
 dump(out/f'exposure_step{step}.json',result);return result
