"""CPU reconstruction and metadata checks only; no forward evaluation or training."""
import hashlib,json,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from rmu_adaptation_utils import load_delta,tensor_hash,materialized
from run_rmu_adaptation import configuration
from omegaconf import OmegaConf
OUT=ROOT/'maintenance/pure_rmu_verification.json'
result={'training_started':False,'full_evaluation_started':False,'device':'cpu','configurations':[],'checkpoints':[]}
def save():OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
def filehash(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
for hook,alpha,lr,s in [(h,a,5e-5,2) for h in [7,12,18] for a in [1,2,4]]+[(12,2,2.5e-5,32),(12,2,5e-5,32)]:
 cfg=configuration(hook,alpha,lr,s,'verification_only',40)
 assert cfg.trainer.handler=='RMU' and cfg.trainer.method_args.retain_loss_type=='EMBED_DIFF'
 assert set(cfg.trainer.method_args)=={'gamma','alpha','retain_loss_type','steering_coeff','module_regex','trainable_params_regex'}
 result['configurations'].append(OmegaConf.to_container(cfg.trainer,resolve=True))
result['file_manifests']=[]
for manifest in ['logs/base_model_manifest.json','logs/rmu_formal_blocks_manifest.json']:
 rows=json.loads((ROOT/manifest).read_text())['files']
 for row in rows:
  p=ROOT/row['path'];assert filehash(p)==row['sha256'],p
 result['file_manifests'].append({'path':manifest,'verified_files':len(rows)})
save()
plan=json.loads((ROOT/'maintenance/pure_rmu_cleanup_plan.json').read_text())
print('Waiting for historical LFS fetch to complete',flush=True)
while any(not (ROOT/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid).exists() for oid in plan['restore_lfs']):
 time.sleep(10)
result['local_lfs_verified']=[]
for oid,row in plan['restore_lfs'].items():
 p=ROOT/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
 assert p.stat().st_size==row['size'] and filehash(p)==oid
 result['local_lfs_verified'].append(oid)
save()
model=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507',dtype=torch.bfloat16,device_map='cpu',local_files_only=True,attn_implementation='sdpa');model.eval()
expected=json.loads((ROOT/'open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text())
assert {n:tensor_hash(p) for n,p in model.named_parameters()}==expected
result['base_all_parameters_verified']=len(expected)
roots=[ROOT/'open-unlearning/saves/unlearn'/p for p in ['rmu_adaptation_v1','rmu_scale_v2','qwen3_4b_rmu_bio_smoke3_v1']]
paths=sorted(p for root in roots for p in root.rglob('checkpoint.json') if 'parameters' in json.loads(p.read_text()))
names={x['name'] for p in paths for x in json.loads(p.read_text())['parameters']};params=dict(model.named_parameters());original={n:params[n].detach().clone() for n in names}
for path in paths:
 with torch.no_grad():
  for n,p in original.items():params[n].copy_(p)
 m=load_delta(model,path.parent,original)
 result['checkpoints'].append({'path':str(path.parent.relative_to(ROOT)),'parameters':len(m['parameters']),'base_compressed_and_reconstructed_sha256_match':True})
 save();print('PASS',path.parent,flush=True)
with torch.no_grad():
 for n,p in original.items():params[n].copy_(p)
assert {n:tensor_hash(p) for n,p in model.named_parameters()}==expected
result['base_restored_after_all_checks']=True
formal=ROOT/'open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_formal80_v1';fp=json.loads((formal/'trained_weight_fingerprints.json').read_text());seen={}
for path in sorted((formal/'model').glob('model-*.safetensors')):
 with safe_open(materialized(path),framework='pt',device='cpu') as f:
  for name in f.keys():
   if name in fp:assert tensor_hash(f.get_tensor(name))==fp[name];seen[name]=True
assert set(seen)==set(fp)
result['formal80_verified_parameters']=len(seen);result['success']=True;save();print('ALL CHECKS PASSED',flush=True)
