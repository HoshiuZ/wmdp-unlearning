"""Finite serial SimNPO/RMU experiment with gradient-calibrated fixed weights."""
import os,sys,json,time,argparse,subprocess,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf,open_dict
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,load_delta,tensor_hash,preserve_rng
from run_rmu_adaptation import configuration,Checkpoints
from run_rmu_formal import Telemetry,fingerprints,now
from rmu_validation import HeldOutValidation
from simnpo_rmu_method import ControlledSimRMU
from model import get_model
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
from trainer import TRAINER_REGISTRY,load_trainer
from trainer.utils import seed_everything
TRAINER_REGISTRY['ControlledSimRMU']=ControlledSimRMU
OUT=REPO/'saves/unlearn/simnpo_rmu_v1'

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['probe','sim','rmu_ce','joint'],required=True);p.add_argument('--sim-scale',type=float,choices=[.5,1.],default=1.);p.add_argument('--check-only',action='store_true');a=p.parse_args();probe=a.variant=='probe';assert a.sim_scale==1. or a.variant=='joint';name='implementation_probe' if probe else a.variant+('_simhalf' if a.sim_scale==.5 else '')+('_minimal' if a.check_only else '_screen40');out=OUT/name
 assert not out.exists();assert shutil.disk_usage(ROOT).free>2*2**30+700*2**20
 out.mkdir(parents=True);cfg=configuration(12,2,5e-5,2,name,40)
 cfg.paths.output_dir=str(out);cfg.trainer.handler='ControlledSimRMU';cfg.trainer.args.output_dir=str(out);cfg.trainer.args.logging_dir=str(out/'logs')
 lock=ROOT/'logs/simnpo_rmu_coefficients.json'
 values={'sim_weight':1.,'ce_weight':1.} if probe else json.loads(lock.read_text())['coefficients'];values['sim_weight']*=a.sim_scale
 with open_dict(cfg.trainer.method_args):
  cfg.trainer.method_args.variant='joint' if probe else a.variant;cfg.trainer.method_args.beta=.1
  for k,v in values.items():cfg.trainer.method_args[k]=v
 dump(out/'training_config.json',OmegaConf.to_container(cfg,resolve=True));run={'name':name,'variant':a.variant,'status':'running','started_at':now(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'data_manifest':'logs/rmu_formal_blocks_manifest.json','beta':.1,'delta':0,'alpha_repr':2,'steering':2,'lr':5e-5,'seed':42,'coefficients':values,'sim_scale':a.sim_scale,'check_only':a.check_only,'max_steps':2 if probe or a.check_only else 40,'space_before':shutil.disk_usage(ROOT)._asdict()};dump(out/'run.json',run)
 for row in json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())['files']:assert __import__('hashlib').sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 seed_everything(42);model,tok=get_model(cfg.model);data=get_data(cfg.data,mode=cfg.mode,tokenizer=tok,template_args=cfg.model.template_args);collator=get_collators(cfg.collator,tokenizer=tok)
 trainer,args=load_trainer(cfg.trainer,model,train_dataset=data['train'],processing_class=tok,data_collator=collator,evaluators=None,template_args=cfg.model.template_args);trainer.create_optimizer();model.to(trainer.accelerator.device)
 selected=[n for n,p in model.named_parameters() if p.requires_grad];assert selected==[f'model.layers.{i}.mlp.down_proj.weight' for i in [10,11,12]];before=fingerprints(model);expected=json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text());assert before==expected;assert not trainer.ref_model.training and not any(p.requires_grad for p in trainer.ref_model.parameters())
 original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected};params=[p for n,p in model.named_parameters() if n in selected]
 # Consume the same control generator RNG as historical RMU, but use the saved direction.
 stored=json.loads((REPO/'saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/control_vector.json').read_text());saved=torch.tensor(stored['values']).reshape(1,1,-1);generator=trainer.get_control_vector
 def fixed_vector(dim):
  generator(dim) # preserve historical one-time RNG consumption at first loss
  trainer.control_vec=saved
  return trainer.control_vec
 trainer.get_control_vector=fixed_vector
 val={r:TokenBlockDataset(hf_args={'path':'json','data_files':str(cfg.validation[r+'_file']),'split':'train'},template_args={},tokenizer=tok,max_length=256) for r in ['forget','retain']}
 start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 if a.check_only:
  model.train();checks=[]
  for i in range(2):
   batch=trainer._prepare_inputs(collator([data['train'][i]]));trainer.optimizer.zero_grad();loss=trainer.compute_loss(model,batch);loss.backward();grads={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if n in selected};assert all(v>0 and __import__('math').isfinite(v) for v in grads.values());assert all(p.grad is None for n,p in model.named_parameters() if n not in selected);assert all(p.grad is None for p in trainer.ref_model.parameters());torch.nn.utils.clip_grad_norm_(params,1.);trainer.optimizer.step();checks.append({'step':i+1,'loss':float(loss.detach()),'gradients':grads})
  after=fingerprints(model);assert set(n for n in before if before[n]!=after[n])==set(selected);run.update(gradient_checks=checks)
 elif probe:
  records=[];model.eval()
  for i in [0,16,32,48]:
   batch=trainer._prepare_inputs(collator([{'forget':val['forget'][i],'retain':val['retain'][i]}]));c,_,_=trainer.components(model,batch);rec={'article_index':i,'losses':{k:float(v.detach()) for k,v in c.items()},'gradients':{}}
   for k in ['rmu_forget','sim_forget','retain_ce']:
    gradients=torch.autograd.grad(c[k],params,retain_graph=True);rec['gradients'][k]=float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt())
   records.append(rec);del c,batch
  means={k:sum(r['gradients'][k] for r in records)/len(records) for k in records[0]['gradients']};assert all(v>0 for v in means.values())
  coeff={'sim_weight':.5*means['rmu_forget']/means['sim_forget'],'ce_weight':.25*means['rmu_forget']/means['retain_ce']};assert all(0<v<100 for v in coeff.values())
  dump(lock,{'coefficients':coeff,'beta':.1,'delta':0,'gradient_probes':records,'mean_gradient_norms':means,'rule':'Original heldout articles0/16/32/48; Sim forget norm target0.5x RMU; CE norm target0.25x RMU. Fixed across all new cases; no test feedback.'});trainer.sim_weight=coeff['sim_weight'];trainer.ce_weight=coeff['ce_weight'];checks=[]
  for variant in ['sim','rmu_ce','joint']:
   with torch.no_grad():
    for n,p in model.named_parameters():
     if n in original:p.copy_(original[n].to(p.device))
   trainer.optimizer=None;trainer.create_optimizer();trainer.variant=variant;model.train()
   for i in range(2):
    batch=trainer._prepare_inputs(collator([data['train'][i]]));trainer.optimizer.zero_grad();loss=trainer.compute_loss(model,batch);loss.backward();grads={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if n in selected};assert all(v>0 and __import__('math').isfinite(v) for v in grads.values());assert all(p.grad is None for n,p in model.named_parameters() if n not in selected);assert all(p.grad is None for p in trainer.ref_model.parameters());torch.nn.utils.clip_grad_norm_(params,1.);trainer.optimizer.step();checks.append({'variant':variant,'step':i+1,'loss':float(loss.detach()),'gradients':grads})
   after=fingerprints(model);assert set(n for n in before if before[n]!=after[n])==set(selected)
  run.update(gradient_checks=checks,calibration=coeff)
 else:
  trainer.add_callback(Telemetry(trainer,out/'training_metrics.jsonl',phase='train'));monitor=HeldOutValidation(trainer,val['forget'],val['retain'],collator,out,every_steps=10);trainer.add_callback(monitor);cp=Checkpoints(trainer,tok,original,out);trainer.add_callback(cp);result=trainer.train();assert trainer.state.global_step==40;trainer.save_state();after=fingerprints(model);assert set(n for n in before if before[n]!=after[n])==set(selected)
  rows=[json.loads(l) for l in (out/'training_metrics.jsonl').open()];tr=[r for r in rows if r['phase']=='train'];assert [r['step'] for r in tr]==list(range(1,41));consumed={r:[i for row in tr for i in row['sampled_'+r+'_indices']] for r in ['forget','retain']};assert consumed==json.loads((REPO/'saves/unlearn/rmu_adaptation_v1/h12_a1_lr5e-05_s2_screen40/consumed_block_indices.json').read_text());dump(out/'consumed_block_indices.json',consumed);dump(out/'component_metrics.json',trainer.component_records);assert len(trainer.component_records)==640
  run.update(training_metrics=result.metrics,development=cp.rows,validation=monitor.results,actual_input_tokens=sum(r['input_tokens'] for r in tr),actual_loss_tokens=sum(r['loss_tokens'] for r in tr),effective_prediction_tokens=sum(r['effective_prediction_tokens'] for r in trainer.component_records))
  assert run['actual_input_tokens']==320619 and run['actual_loss_tokens']==319339
 assert fingerprints(trainer.ref_model)==before;dump(out/'control_vector.json',{'values':saved.flatten().tolist(),'sha256':tensor_hash(saved),'l2_norm':float(saved.norm())});run.update(status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-start,selected_parameters=selected,frozen_parameters_unchanged=True,reference_unchanged=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),space_after=shutil.disk_usage(ROOT)._asdict());dump(out/'run.json',run);print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
