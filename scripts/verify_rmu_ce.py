"""Verify retained artifact hashes, exact reconstruction and optional migration equivalence."""
import argparse,json,gzip,importlib.util
from pathlib import Path
import torch
from rmu_ce_utils import ROOT,OUT,BASE,dump,digest,materialized,load_delta,tensor_hash,base_samples
from run_rmu_ce import build
from rmu_ce_callbacks import fingerprints
from rmu_ce_method import RMUCE

def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-source',type=Path);p.add_argument('--output',default='checks/verification');p.add_argument('--deterministic-math',action='store_true');a=p.parse_args();
 if a.deterministic_math:
  torch.use_deterministic_algorithms(True);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 out=(OUT/a.output).resolve();assert out.is_relative_to(OUT.resolve());assert not out.exists();cfg,t,tok,data,col,before=build(42,out);model=t.model;selected=[n for n,p in model.named_parameters() if p.requires_grad];original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected};report={'deterministic_math_attention':a.deterministic_math,'selected_parameters':selected,'original_fingerprint_verified':True,'checkpoints':[]}
 if a.legacy_source:
  spec=importlib.util.spec_from_file_location('migration_legacy',a.legacy_source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);t.variant='rmu_ce';t.sim_weight=.1566522562277117;t.beta=.1;t.gradient_cap=False;model.train();comparisons=[]
  for i in range(2):
   batch=t._prepare_inputs(col([data['train'][i]]));t.__class__=mod.ControlledSimRMU;t.optimizer.zero_grad();old_loss=t.compute_loss(model,batch);old_loss.backward();grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if n in selected};old_value=float(old_loss.detach());del old_loss
   t.__class__=RMUCE;t.optimizer.zero_grad();new_loss=t.compute_loss(model,batch);new_loss.backward();equal={n:torch.equal(grads[n],p.grad.detach().cpu()) for n,p in model.named_parameters() if n in selected};assert old_value==float(new_loss.detach()) and all(equal.values()),'Loss or gradients changed during refactor';assert all(p.grad is None for n,p in model.named_parameters() if n not in selected);assert all(p.grad is None for p in t.ref_model.parameters());torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);t.optimizer.step();comparisons.append({'step':i+1,'loss':old_value,'gradients_bitwise_equal':equal});del grads,new_loss
  report['legacy_source_sha256']=digest(a.legacy_source.read_bytes());report['migration_equivalence']=comparisons
 with torch.no_grad():
  for n,p in model.named_parameters():
   if n in original:p.copy_(original[n].to(p.device))
 assert fingerprints(model)==before and fingerprints(t.ref_model)==before
 for seed in [42,43]:
  for step in [10,20,40]:
   cp=OUT/f'seed{seed}/checkpoint-{step}';m=load_delta(model,cp,original);report['checkpoints'].append({'seed':seed,'step':step,'trained_tensor_sha_verified':True,'compressed_sha_verified':True})
  dest=OUT/f'full_evaluations/seed{seed}_checkpoint-40';d=json.loads((dest/'samples.json').read_text());result=json.loads((dest/'result.json').read_text())
  for task,subjects in base_samples().items():
   ref={(s,x['doc_id']):x for s,rows in subjects.items() for x in rows};assert len(d[task])==len(ref) and {(x['task'],x['doc_id']) for x in d[task]}==set(ref)
   for x in d[task]:
    y=ref[x['task'],x['doc_id']];assert x['doc_hash']==y['doc_hash'] and x['prompt_hash']==y['prompt_hash'] and x['target']==int(y['doc']['answer'])
   assert sum(x['acc'] for x in d[task])==result['metrics'][task]['correct']
 report.update(full_request_hashes_and_counts_verified=True,reference_unchanged=True);dump(out/'report.json',report);dump(ROOT/'logs/rmu_ce_verification.json',report);print('VERIFIED',json.dumps(report),flush=True)
if __name__=='__main__':main()
