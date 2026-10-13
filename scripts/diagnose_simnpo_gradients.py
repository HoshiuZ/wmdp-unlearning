"""No-update gradient geometry; original reference activations cached in RAM."""
import json,time,itertools,subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors import safe_open
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,tensor_hash,load_delta
from simnpo_rmu_method import token_nll,simnpo
from run_rmu_formal import fingerprints
from data.pretraining import TokenBlockDataset
from trainer.unlearn.rmu import RMU
OUT=REPO/'saves/unlearn/simnpo_rmu_v1';torch.set_num_threads(8)
KEYS=['rmu_forget','sim_forget','retain_repr','retain_ce'];PAIRS=list(itertools.combinations(KEYS,2))
def main():
 start=time.perf_counter();coeff=json.loads((ROOT/'logs/simnpo_rmu_coefficients.json').read_text())['coefficients'];weights={'rmu_forget':1.,'sim_forget':coeff['sim_weight'],'retain_repr':2.,'retain_ce':coeff['ce_weight']}
 model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model.eval();before=fingerprints(model);assert before==json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text())
 names=[f'model.layers.{i}.mlp.down_proj.weight' for i in range(model.config.num_hidden_layers)];selected=names[10:13];parameters=dict(model.named_parameters());original={n:parameters[n].detach().cpu().clone() for n in selected}
 for n,p in parameters.items():p.requires_grad_(n in names)
 pool=[parameters[n] for n in names];module=model.model.layers[12];control=torch.tensor(json.loads((OUT/'joint_screen40/control_vector.json').read_text())['values']).reshape(1,1,-1).cuda().bfloat16()
 datasets={r:TokenBlockDataset(hf_args={'path':'json','data_files':str(ROOT/f'data/processed/bio_v1/formal_blocks_v1/{r}/validation.jsonl'),'split':'train'},template_args={},tokenizer=tok,max_length=256) for r in ['forget','retain']}
 def batch(role,i):return {k:v.unsqueeze(0).cuda() for k,v in datasets[role][i].items() if k in ['input_ids','attention_mask','labels']}
 def forward(b):
  cache=[];hook=module.register_forward_hook(lambda m,a,o:cache.append(o[0] if isinstance(o,tuple) else o))
  try:o=model(**b)
  finally:hook.remove()
  return cache[0],o
 reference={}
 with torch.no_grad():
  for i in [0,16,32,48]:reference[i]=forward(batch('retain',i))[0].detach().cpu()
 results={}
 for label,path in [('original',None),('rmu_ce40',OUT/'rmu_ce_screen40/checkpoint-40'),('joint20',OUT/'joint_screen40/checkpoint-20'),('joint40',OUT/'joint_screen40/checkpoint-40')]:
  with torch.no_grad():
   for n in selected:parameters[n].copy_(original[n].cuda())
  if path:load_delta(model,path,original)
  records=[]
  for i in [0,16,32,48]:
   f=batch('forget',i);r=batch('retain',i);fa,fo=forward(f);ra,ro=forward(r)
   losses={'rmu_forget':RMU.compute_activation_loss(None,fa,control.expand_as(fa),f['labels']!=-100),'sim_forget':simnpo(fo.logits,f['labels'],beta=.1),'retain_repr':RMU.compute_activation_loss(None,ra,reference[i].to(ra),r['labels']!=-100),'retain_ce':token_nll(ro.logits,r['labels']).mean()};grads={}
   for k in KEYS:
    g=torch.autograd.grad(losses[k],pool,retain_graph=True,allow_unused=True);grads[k]=[x.detach().cpu() if x is not None else None for x in g];del g
   rec={'article_index':i,'losses':{k:float(v.detach()) for k,v in losses.items()},'layers':{}};aggregate={group:{'squared_norms':{k:0. for k in KEYS},'dots':{a+'|'+b:0. for a,b in PAIRS}} for group in ['selected','outside','all']}
   for layer,n in enumerate(names):
    vectors={k:grads[k][layer].float().flatten() if grads[k][layer] is not None else None for k in KEYS};sq={k:float(torch.dot(v,v)) if v is not None else 0. for k,v in vectors.items()};dots={a+'|'+b:float(torch.dot(vectors[a],vectors[b])) if vectors[a] is not None and vectors[b] is not None else 0. for a,b in PAIRS}
    if layer>12:assert sq['rmu_forget']==0 and sq['retain_repr']==0
    rec['layers'][str(layer)]={'squared_norms':sq,'dots':dots}
    for group in ['all','selected' if n in selected else 'outside']:
     for k in KEYS:aggregate[group]['squared_norms'][k]+=sq[k]
     for k in dots:aggregate[group]['dots'][k]+=dots[k]
   for group,v in aggregate.items():
    v['weighted_norms']={k:weights[k]*v['squared_norms'][k]**.5 for k in KEYS};v['cosines']={a+'|'+b:v['dots'][a+'|'+b]/(v['squared_norms'][a]*v['squared_norms'][b])**.5 if v['squared_norms'][a]*v['squared_norms'][b]>0 else None for a,b in PAIRS}
    dot=lambda a,b:v['squared_norms'][a] if a==b else v['dots']['|'.join(sorted([a,b],key=KEYS.index))]
    fg=['rmu_forget','sim_forget'];rt=['retain_repr','retain_ce'];fs=sum(weights[a]*weights[b]*dot(a,b) for a in fg for b in fg);rs=sum(weights[a]*weights[b]*dot(a,b) for a in rt for b in rt);cross=sum(weights[a]*weights[b]*dot(a,b) for a in fg for b in rt)
    v['forget_retain_cosine']=cross/(fs*rs)**.5 if fs*rs>0 else None;v['sim_over_rmu_weighted_norm']=v['weighted_norms']['sim_forget']/v['weighted_norms']['rmu_forget'] if v['weighted_norms']['rmu_forget'] else None
   rec['groups']=aggregate;records.append(rec);del losses,grads,fa,fo,ra,ro,vectors
   print('PROBE',label,i,aggregate['selected']['sim_over_rmu_weighted_norm'],aggregate['selected']['forget_retain_cosine'],flush=True)
  results[label]=records;dump(ROOT/'logs/simnpo_rmu_gradient_diagnostics.partial.json',{'weights':weights,'results':results,'incomplete':True})
 with torch.no_grad():
  for n in selected:parameters[n].copy_(original[n].cuda())
 assert fingerprints(model)==before
 dump(ROOT/'logs/simnpo_rmu_gradient_diagnostics.json',{'weights':weights,'results':results,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'seconds':time.perf_counter()-start,'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'no_optimizer_steps':True,'original_restored_fingerprints_equal':True,'reference':'Original frozen eval no_grad hook12 activations cached in RAM; no second model or saved activations','scope':f'{model.config.num_hidden_layers} down_proj gradients only, not all model parameters','limitation':'Four heldout article probes; outside gradient mass is not proof of capacity bottleneck or improved expanded training.'})
if __name__=='__main__':main()
