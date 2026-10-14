"""Exact BF16 XOR checkpoints, frozen development split and protocol-faithful scoring."""
import os,sys,json,hashlib,gzip,subprocess,random,contextlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'open-unlearning';BASE=ROOT/'models/Qwen3-4B-Instruct-2507';OUT=REPO/'saves/unlearn/rmu_ce_v1'
sys.path.insert(0,str(REPO/'src'))
from lm_eval.models.hf_vlms import HFLM
from lm_eval.api.instance import Instance

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(x):return hashlib.sha256(x).hexdigest()
def tensor_hash(p):return digest(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
def base_samples():
 return {task:json.loads((REPO/'saves/eval'/run/'LMEval_EVAL.json').read_text())[task] for task,run in [('wmdp_bio','qwen3_4b_base_bio_full_v1'),('mmlu','qwen3_4b_base_mmlu_0shot_full_v1')]}
def prepare_dev():
 p=ROOT/'reference/rmu_ce/dev_split.json'
 if p.exists():return json.loads(p.read_text())
 selections={}; stats={}
 for task,subjects in base_samples().items():
  selections[task]={};n=correct=0
  for name,rows in subjects.items():
   chosen=sorted(rows,key=lambda s:digest(f'adapt-v1:42:{name}:{s["doc_id"]}'.encode()))[:256 if task=='wmdp_bio' else 30]
   selections[task][name]=sorted(s['doc_id'] for s in chosen);n+=len(chosen);correct+=sum(s['acc'] for s in chosen)
  stats[task]={'n':n,'correct':correct,'accuracy_percent':100*correct/n}
 result={'seed':42,'selection':'Smallest SHA256(adapt-v1:42:task:doc_id); 256 Bio and 30 per MMLU subject','independent_validation':False,'limitation':'Subset of previously evaluated official tests. All adaptation full-set scores are exploratory, not independent held-out tests. Retain training data is not this validation set.','selected_doc_ids':selections,'baseline':stats}
 dump(p,result);return result
@contextlib.contextmanager
def preserve_rng():
 py=random.getstate();npstate=np.random.get_state();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
 try:yield
 finally:random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(cuda)
def score(model,tokenizer,out,full=False):
 from time import perf_counter
 start=perf_counter();dev=prepare_dev();data=base_samples();model.eval();rows={};summary={}
 wrapper=HFLM(model,tokenizer=tokenizer,batch_size=1)
 with preserve_rng():
  for task,subjects in data.items():
   chosen=[];requests=[]
   for name,records in subjects.items():
    allowed=set(dev['selected_doc_ids'][task][name])
    for sample in records:
     if not full and sample['doc_id'] not in allowed:continue
     chosen.append((name,sample))
     for i,arg in enumerate(sample['arguments']):requests.append(Instance(request_type='loglikelihood',doc=sample['doc'],arguments=tuple(arg),idx=i))
   responses=wrapper.loglikelihood(requests);result=[];baseline_correct=correct=0
   for j,(name,sample) in enumerate(chosen):
    values=[float(responses[4*j+k][0]) for k in range(4)];prediction=int(np.argmax(values));target=int(sample['doc']['answer']);acc=int(prediction==target)
    result.append({'task':name,'doc_id':sample['doc_id'],'doc_hash':sample['doc_hash'],'prompt_hash':sample['prompt_hash'],'target':target,'prediction':prediction,'scores':values,'acc':acc,'baseline_acc':int(sample['acc'])})
    correct+=acc;baseline_correct+=int(sample['acc'])
   rows[task]=result;n=len(result)
   summary[task]={'n':n,'correct':correct,'accuracy_percent':100*correct/n,'baseline_accuracy_percent':100*baseline_correct/n,'change_percentage_points':100*(correct-baseline_correct)/n}
 dump(out/'samples.json',rows);dump(out/'summary.json',{'full':full,'exploratory':True,'protocol':'Archived exact baseline loglikelihood requests, BF16/SDPA,batch1,no chat/system,0shot; deterministic argmax; no generation','seconds':perf_counter()-start,'metrics':summary})
 return summary

def save_delta(model,original,path):
 """Store exact uint16 XOR against pinned original, gzip, then retain LFS pointer."""
 path.mkdir(parents=True,exist_ok=False);manifest={'format':'bf16-uint16-xor-gzip-v1','base_manifest':'logs/base_model_manifest.json','parameters':[]}
 for name,p in model.named_parameters():
  if name not in original:continue
  a=original[name].contiguous().view(torch.uint16).numpy();b=p.detach().cpu().contiguous().view(torch.uint16).numpy()
  payload=np.bitwise_xor(a,b).tobytes();file=path/(name+'.xor.gz')
  external=os.getenv('RMU_CE_LFS_STORAGE')
  if external:
   import tempfile
   cache=Path(external).resolve();cache.mkdir(parents=True,exist_ok=True)
   descriptor,temporary=tempfile.mkstemp(suffix='.xor.gz',dir=cache);os.close(descriptor);compressed=Path(temporary)
  else:cache=ROOT/'.git/lfs';compressed=file
  with gzip.open(compressed,'wb',compresslevel=6) as f:f.write(payload)
  compressed_sha=digest(compressed.read_bytes());size=compressed.stat().st_size
  command=['git']+(['-c',f'lfs.storage={cache}'] if external else [])+['lfs','clean','--',str(file.relative_to(ROOT))]
  with compressed.open('rb') as f:pointer=subprocess.check_output(command,stdin=f,cwd=ROOT)
  assert pointer.startswith(b'version https://git-lfs.github.com/spec/v1\n');file.write_bytes(pointer)
  if external:compressed.unlink()
  manifest['parameters'].append({'name':name,'shape':list(p.shape),'dtype':'bfloat16','file':file.name,'bytes':size,'compressed_sha256':compressed_sha,'base_tensor_sha256':tensor_hash(original[name]),'trained_tensor_sha256':tensor_hash(p),'local_lfs_cache':str(cache)})
 dump(path/'checkpoint.json',manifest);return manifest

def materialized(path):
 with path.open('rb') as f:head=f.read(150)
 if not head.startswith(b'version https://git-lfs.github.com/spec/v1'):return path
 oid=[l.split(':',1)[1] for l in head.decode().splitlines() if l.startswith('oid sha256:')][0]
 default=ROOT/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
 if default.exists():return default
 manifest=path.parent/'checkpoint.json'
 if manifest.exists():
  row=next(r for r in json.loads(manifest.read_text())['parameters'] if r['file']==path.name)
  alternative=Path(row.get('local_lfs_cache',str(ROOT/'.git/lfs')))/'objects'/oid[:2]/oid[2:4]/oid
  if alternative.exists():return alternative
 raise FileNotFoundError(f'Git LFS object {oid} unavailable; fetch/materialize the checkpoint first')

def load_delta(model,path,original=None):
 manifest=json.loads((path/'checkpoint.json').read_text());params=dict(model.named_parameters())
 if original is None:
  expected=json.loads((ROOT/'reference/rmu_ce/base_weight_fingerprints.json').read_text())
  assert {n:tensor_hash(p) for n,p in model.named_parameters()}==expected,'Complete starting model differs from pinned original'
 for row in manifest['parameters']:
  p=params[row['name']];a=original[row['name']] if original is not None else p.detach().cpu()
  assert tensor_hash(a)==row['base_tensor_sha256']
  file=materialized(path/row['file']);assert digest(file.read_bytes())==row['compressed_sha256']
  delta=np.frombuffer(gzip.decompress(file.read_bytes()),dtype=np.uint16).reshape(row['shape'])
  restored=np.bitwise_xor(a.contiguous().view(torch.uint16).numpy(),delta).copy()
  t=torch.from_numpy(restored).view(torch.bfloat16);assert tensor_hash(t)==row['trained_tensor_sha256']
  with torch.no_grad():p.copy_(t.to(p.device))
 return manifest
