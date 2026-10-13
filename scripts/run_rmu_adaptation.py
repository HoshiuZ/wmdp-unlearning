"""Bounded Qwen RMU screening; fresh original weights, exact deltas, frozen dev set."""
import os,sys,json,time,subprocess,gc,argparse
from pathlib import Path
import torch
from transformers import TrainerCallback
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from rmu_adaptation_utils import ROOT,REPO,BASE,OUT,dump,tensor_hash,score,save_delta,load_delta,preserve_rng,prepare_dev
os.chdir(REPO)
from model import get_model
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
from trainer import load_trainer
from trainer.utils import seed_everything
from run_rmu_formal import Telemetry,fingerprints,now
from rmu_validation import HeldOutValidation

class Checkpoints(TrainerCallback):
 def __init__(self,trainer,tokenizer,original,out):self.trainer=trainer;self.tok=tokenizer;self.original=original;self.out=out;self.rows=[]
 def on_step_end(self,args,state,control,**kwargs):
  if state.global_step not in (10,20,40):return
  model=self.trainer.model;was=model.training;start=time.perf_counter()
  with preserve_rng():
   path=self.out/f'checkpoint-{state.global_step}'
   save_delta(model,self.original,path)
   # Load from original values then compare exact trained hashes before evaluation.
   expected={n:tensor_hash(p) for n,p in model.named_parameters() if n in self.original}
   load_delta(model,path,self.original)
   assert expected=={n:tensor_hash(p) for n,p in model.named_parameters() if n in self.original}
   metrics=score(model,self.tok,path/'development')
  model.train(was)
  self.rows.append({'step':state.global_step,'metrics':metrics,'checkpoint':str(path.relative_to(ROOT)),'save_reload_identical':True,'seconds':time.perf_counter()-start})
  dump(self.out/'development_results.json',self.rows)
  print('DEV',state.global_step,json.dumps(metrics),flush=True)

def configuration(hook,alpha,lr,steering,name,steps):
 with initialize_config_dir(config_dir=str(REPO/'configs'),version_base=None):cfg=compose(config_name='train',overrides=['experiment=unlearn/wmdp/qwen3_4b_bio_formal'])
 cfg.task_name=name;cfg.paths.work_dir=str(REPO);cfg.trainer.args.max_steps=steps
 cfg.trainer.args.learning_rate=lr;cfg.trainer.args.gradient_accumulation_steps=16
 cfg.trainer.method_args.alpha=alpha;cfg.trainer.method_args.steering_coeff=steering
 cfg.trainer.method_args.module_regex=rf'model\.layers\.{hook}'
 cfg.trainer.method_args.trainable_params_regex=[rf'model\.layers\.({hook-2}|{hook-1}|{hook})\.mlp\.down_proj\.weight']
 cfg.trainer.args.output_dir=str(OUT/name);cfg.trainer.args.logging_dir=str(OUT/name/'logs')
 cfg.validation.every_steps=10;cfg.validation.selection_rule='development constrained ranking; exploratory'
 return cfg

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--hook',type=int,choices=[7,12,18],required=True);parser.add_argument('--alpha',type=float,default=1);parser.add_argument('--lr',type=float,default=5e-5);parser.add_argument('--steering',type=float,default=2);parser.add_argument('--minimal',action='store_true');a=parser.parse_args()
 name=f'h{a.hook}_a{a.alpha:g}_lr{a.lr:g}_s{a.steering:g}'+('_minimal' if a.minimal else '_screen40')
 out=OUT/name
 if out.exists():raise RuntimeError(f'Refuse to overwrite {out}')
 out.mkdir(parents=True);cfg=configuration(a.hook,a.alpha,a.lr,a.steering,name,40)
 dump(out/'training_config.json',OmegaConf.to_container(cfg,resolve=True))
 run={'name':name,'started_at':now(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':tensorless_hash(Path(__file__)), 'hook':a.hook,'alpha':a.alpha,'learning_rate':a.lr,'steering_coefficient':a.steering,'seed':42,'minimal':a.minimal,'status':'running','command':' '.join(sys.argv),'data_manifest':'logs/rmu_formal_blocks_manifest.json'}
 dump(out/'run.json',run)
 for row in json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())['files']:assert tensorless_hash(ROOT/row['path'])==row['sha256']
 seed_everything(42);model,tok=get_model(cfg.model);data=get_data(cfg.data,mode=cfg.mode,tokenizer=tok,template_args=cfg.model.template_args);collator=get_collators(cfg.collator,tokenizer=tok)
 trainer,args=load_trainer(trainer_cfg=cfg.trainer,model=model,train_dataset=data['train'],processing_class=tok,data_collator=collator,evaluators=None,template_args=cfg.model.template_args);trainer.create_optimizer()
 selected=[n for n,p in model.named_parameters() if p.requires_grad];assert selected==[f'model.layers.{i}.mlp.down_proj.weight' for i in range(a.hook-2,a.hook+1)]
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)==74711040
 assert not trainer.ref_model.training and not any(p.requires_grad for p in trainer.ref_model.parameters())
 before=fingerprints(model);baseline=json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text());assert before==baseline
 original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected}
 torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 if a.minimal:
  model.train();gradchecks=[]
  for i in range(2):
   batch=trainer._prepare_inputs(collator([data['train'][i]]));trainer.optimizer.zero_grad()
   with trainer.compute_loss_context_manager():loss=trainer.compute_loss(model,batch)
   loss.backward();grads={n:float(p.grad.detach().float().norm()) for n,p in model.named_parameters() if n in selected}
   assert all(v>0 and torch.isfinite(torch.tensor(v)) for v in grads.values());assert all(p.grad is None for n,p in model.named_parameters() if n not in selected);assert all(p.grad is None for p in trainer.ref_model.parameters())
   act,_=trainer.forward_with_cache(trainer.ref_model,{k:v for k,v in batch['retain'].items() if k in ('input_ids','attention_mask','labels')},trainer.ref_module,no_grad=True);assert not act.requires_grad
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);trainer.optimizer.step();gradchecks.append({'step':i+1,'loss':float(loss),'gradient_norms':grads})
  after=fingerprints(model);changed=[n for n in before if before[n]!=after[n]];assert set(changed)==set(selected)
  assert fingerprints(trainer.ref_model)==before
  run.update(gradient_checks=gradchecks,changed_parameters=changed,reference_fingerprints_unchanged=True,reference_activation_requires_grad=False,all_selected_gradients_nonzero=True,frozen_parameters_unchanged=True)
 else:
  trainer.add_callback(Telemetry(trainer,out/'training_metrics.jsonl',phase='train'))
  val={role:TokenBlockDataset(hf_args={'path':'json','data_files':str(cfg.validation[f'{role}_file']),'split':'train'},template_args=cfg.model.template_args,tokenizer=tok,max_length=256) for role in ('forget','retain')}
  monitor=HeldOutValidation(trainer,val['forget'],val['retain'],collator,out,every_steps=10);trainer.add_callback(monitor)
  checkpoints=Checkpoints(trainer,tok,original,out);trainer.add_callback(checkpoints)
  result=trainer.train();assert trainer.state.global_step==40;trainer.save_state()
  after=fingerprints(model);changed=[n for n in before if before[n]!=after[n]];assert set(changed)==set(selected)
  assert fingerprints(trainer.ref_model)==before
  records=[json.loads(l) for l in (out/'training_metrics.jsonl').open()];train=[r for r in records if r['phase']=='train'];assert [r['step'] for r in train]==list(range(1,41))
  consumed={role:[i for r in train for i in r[f'sampled_{role}_indices']] for role in ('forget','retain')};dump(out/'consumed_block_indices.json',consumed)
  control=trainer.control_vec.detach().cpu();dump(out/'control_vector.json',{'values':control.flatten().tolist(),'sha256':tensor_hash(control),'l2_norm':float(control.norm()),'fixed':True})
  run.update(training_metrics=result.metrics,development=checkpoints.rows,validation=monitor.results,changed_parameters=changed,frozen_parameters_unchanged=True,reference_fingerprints_unchanged=True,actual_exposures={r:len(v) for r,v in consumed.items()},actual_unique_blocks={r:len(set(v)) for r,v in consumed.items()},control_vector_sha256=tensor_hash(control))
  # Exact current-configuration control must preserve first40 sampling and control direction.
  if a.hook==7 and a.alpha==1 and a.lr==5e-5 and a.steering==2:
   old=REPO/'saves/unlearn/qwen3_4b_rmu_bio_formal80_v1'
   formal=json.loads((old/'consumed_block_indices.json').read_text());assert all(consumed[r]==formal[r][:640] for r in consumed),'sampling drift from current control'
   assert tensor_hash(torch.load(old/'control_vector.pt',weights_only=True))==tensor_hash(control),'control direction drift'
   run['current_control_first40_sampling_identical']=True
 run.update(status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-start,selected_parameters=selected,trainable_parameters=74711040,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 dump(out/'run.json',run);print('RUN_COMPLETE',name,run['cuda_peak_allocated_bytes'],flush=True)
def tensorless_hash(path):
 import hashlib
 return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':main()
