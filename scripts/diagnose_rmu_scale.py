"""Held-out hook12 diagnostics of original and preserved alpha1/alpha2 checkpoints."""
import os,json,time,subprocess
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,OUT,dump,load_delta,tensor_hash
sys_path=ROOT/'scripts'
import sys
sys.path.insert(0,str(REPO/'src'));os.chdir(REPO)
from data.pretraining import TokenBlockDataset
from data.collators import DataCollatorForSupervisedDataset
REPORT=ROOT/'logs/rmu_scale_v2_diagnostics.json'
def stats(values):
 x=np.concatenate(values);return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'quantiles':dict(zip(['min','p10','p25','p50','p75','p90','p99','max'],map(float,np.quantile(x,[0,.1,.25,.5,.75,.9,.99,1]))))}
def activation(model,batch):
 cache=[];hook=model.model.layers[12].register_forward_hook(lambda m,i,o:cache.append(o[0] if isinstance(o,tuple) else o))
 try:model.model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'],use_cache=False)
 finally:hook.remove()
 return cache[0]
def main():
 if REPORT.exists():raise RuntimeError('Refuse to overwrite diagnosis')
 start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');model.eval();model.requires_grad_(False)
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);collator=DataCollatorForSupervisedDataset(tok,padding_side='left',index='index')
 data={role:TokenBlockDataset(hf_args={'path':'json','data_files':str(ROOT/f'data/processed/bio_v1/formal_blocks_v1/{role}/validation.jsonl'),'split':'train'},template_args={},tokenizer=tok,max_length=256) for role in ['forget','retain']}
 original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in [f'model.layers.{i}.mlp.down_proj.weight' for i in [10,11,12]]}
 reference={r:[] for r in data};results={}
 checkpoints=[('original',None,1),('alpha1',OUT/'h12_a1_lr5e-05_s2_screen40',1),('alpha2',OUT/'h12_a2_lr5e-05_s2_screen40',2)]
 control_records={name:json.loads((run/'control_vector.json').read_text()) for name,run,a in checkpoints if run is not None}
 control=torch.tensor(control_records['alpha1']['values']).reshape(1,1,-1)
 assert tensor_hash(control)==control_records['alpha1']['sha256']==control_records['alpha2']['sha256']
 assert abs(float(control.norm())-2)<1e-6
 selected=[p for n,p in model.named_parameters() if n in original]
 for name,run,alpha in checkpoints:
  if run:load_delta(model,run/'checkpoint-40',original)
  norms={r:[] for r in data};cos_original={r:[] for r in data};cos_target={r:[] for r in data};ratios={r:[] for r in data};metrics={r:[] for r in data};native={r:[] for r in data};radial={r:[] for r in data};angular={r:[] for r in data}
  for role,ds in data.items():
   for i in range(len(ds)):
    b={k:v.to('cuda') for k,v in collator([ds[i]]).items()};mask=b['attention_mask'].bool()&(b['labels']!=-100)
    with torch.no_grad():h=activation(model,b)
    if name=='original':reference[role].append(h.detach().cpu())
    ref=reference[role][i].to('cuda');v=h.float()[mask];v0=ref.float()[mask];target=control.to('cuda').float().expand_as(h)[mask]
    hn=v.norm(dim=-1);rn=v0.norm(dim=-1);ct=torch.nn.functional.cosine_similarity(v,target);c0=torch.nn.functional.cosine_similarity(v,v0)
    norms[role].append(hn.cpu().numpy());cos_target[role].append(ct.cpu().numpy());cos_original[role].append(c0.cpu().numpy());ratios[role].append((hn/rn).cpu().numpy())
    dest=target if role=='forget' else v0;metrics[role].append(float((v-dest).square().mean()))
    # Native BF16 reduction matches actual RMU; FP32 diagnostics separately reported.
    target_native=control.to(device='cuda',dtype=h.dtype).expand_as(h) if role=='forget' else ref
    native[role].append(float(((h-target_native).square()*mask.unsqueeze(-1)).mean(dim=2).sum()/mask.sum()))
    radial[role].append(float((hn-rn).square().mean()/h.shape[-1]));angular[role].append(float((2*hn*rn*(1-c0)).mean()/h.shape[-1]))
  gradients=[]
  for p in selected:p.requires_grad_(True)
  for i in [0,16,32,48]:
   g={}
   for role in data:
    b={k:v.to('cuda') for k,v in collator([data[role][i]]).items()};mask=b['attention_mask'].bool()&(b['labels']!=-100);h=activation(model,b)
    dest=control.to(device='cuda',dtype=h.dtype).expand_as(h) if role=='forget' else reference[role][i].to('cuda')
    loss=((h-dest).square()*mask.unsqueeze(-1)).mean(dim=2).sum()/mask.sum()
    grads=torch.autograd.grad(loss,selected);g[role]=float(torch.sqrt(sum(z.detach().float().square().sum() for z in grads)))
    del grads,h,loss
   gradients.append({'article_index':i,'forget_gradient_norm':g['forget'],'retain_gradient_norm':g['retain'],'alpha_weighted_retain_gradient_norm':alpha*g['retain'],'forget_over_alpha_retain_ratio':g['forget']/(alpha*g['retain']) if g['retain']>0 else None})
  for p in selected:p.requires_grad_(False)
  results[name]={'alpha':alpha,'roles':{r:{'l2_norm':stats(norms[r]),'cosine_to_original':stats(cos_original[r]),'cosine_to_saved_target':stats(cos_target[r]),'norm_ratio_to_original':stats(ratios[r]),'article_mean_mse_fp32':float(np.mean(metrics[r])),'article_mean_mse_native_bf16':float(np.mean(native[r])),'radial_change_mse_fp32':float(np.mean(radial[r])),'angular_change_mse_fp32':float(np.mean(angular[r]))} for r in data},'weighted_losses_native_bf16':{'forget':float(np.mean(native['forget'])),'alpha_retain':alpha*float(np.mean(native['retain'])),'total':float(np.mean(native['forget']))+alpha*float(np.mean(native['retain']))},'gradient_probes':gradients}
  print('DIAG',name,{r:results[name]['roles'][r]['l2_norm']['mean'] for r in data},gradients,flush=True)
 report={'model':'Qwen3-4B-Instruct-2507','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'hook':12,'layer_indexing':'zero-based decoder block output','mask':'attention_mask AND labels!=-100; excludes padding and existing first-token loss mask','validation_blocks_per_role':64,'control_vector_source':str((OUT/'h12_a1_lr5e-05_s2_screen40/control_vector.json').relative_to(ROOT)),'control_vector_sha256':tensor_hash(control),'control_l2':float(control.norm()),'results':results,'seconds':time.perf_counter()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'no_weights_written':True,'limitation':'Held-out articles for representation diagnostics; no knowledge-deletion inference. Probe gradients eval-mode, separate losses, no optimizer updates.'}
 dump(REPORT,report)
if __name__=='__main__':main()
