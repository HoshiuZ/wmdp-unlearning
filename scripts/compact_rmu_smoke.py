import sys,json,hashlib,shutil
from pathlib import Path
import torch
from safetensors import safe_open
sys.path.insert(0,str(Path.cwd()/'scripts'))
from rmu_adaptation_utils import ROOT,BASE,save_delta,load_delta,tensor_hash,dump
folder=ROOT/'open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1';modeldir=folder/'model';target=folder/'delta_checkpoint'
before=json.loads((folder/'base_weight_fingerprints.json').read_text());after=json.loads((folder/'trained_weight_fingerprints.json').read_text());changed=[n for n in before if before[n]!=after[n]]
assert changed==[f'model.layers.{i}.mlp.down_proj.weight' for i in [5,6,7]]
assert not target.exists()
index=json.loads((modeldir/'model.safetensors.index.json').read_text())['weight_map'];actual={};trained={}
for filename in sorted(set(index.values())):
 with safe_open(modeldir/filename,framework='pt',device='cpu') as f:
  for name in f.keys():
   tensor=f.get_tensor(name);actual[name]=tensor_hash(tensor)
   if name in changed:trained[name]=tensor.clone()
 print('VERIFIED_SHARD',filename,flush=True)
assert actual==after,'Complete smoke parameter fingerprints differ'
baseindex=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'];original={}
for name in changed:
 with safe_open(BASE/baseindex[name],framework='pt',device='cpu') as f:original[name]=f.get_tensor(name).clone()
 assert tensor_hash(original[name])==before[name]
class Selected:
 def named_parameters(self):return iter(trained.items())
model=Selected();manifest=save_delta(model,original,target)
for name in changed:trained[name]=original[name].clone()
load_delta(model,target,original=original)
assert all(tensor_hash(trained[n])==after[n] for n in changed)
assert all(actual[n]==before[n] for n in actual if n not in changed)
artifacts=json.loads((folder/'model_artifacts.json').read_text());removed=[]
for filename in sorted(set(index.values())):
 p=modeldir/filename;entry=next(r for r in artifacts['files'] if Path(r['path']).name==filename)
 assert p.stat().st_size==entry['size_bytes'];h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 assert h.hexdigest()==entry['sha256'];removed.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
record={'scope':'Lossless smoke full weights compacted to BF16 XOR checkpoint; base/formal and all old experiment deltas retained','complete_smoke_fingerprints_verified':True,'unchanged_parameters_match_base':True,'reconstruction_bit_exact':True,'checkpoint':str(target.relative_to(ROOT)),'delta_bytes':sum(p['bytes'] for p in manifest['parameters']),'removed_full_shards':removed,'space_before':shutil.disk_usage(ROOT)._asdict()}
dump(ROOT/'logs/smoke_compaction_20261014.json',record)
for item in removed:(ROOT/item['path']).unlink()
record['space_after']=shutil.disk_usage(ROOT)._asdict();dump(ROOT/'logs/smoke_compaction_20261014.json',record);print('COMPACTION_COMPLETE',record,flush=True)
