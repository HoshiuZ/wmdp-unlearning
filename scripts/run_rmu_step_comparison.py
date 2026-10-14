"""Locked continuous120 pure-RMU run. No optimizer-state resume is supported."""
import argparse,copy,gc,hashlib,importlib.metadata,json,math,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
from transformers import TrainerCallback
from omegaconf import OmegaConf
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,score,save_delta,load_delta,tensor_hash
from rmu_step_common import HISTORY,RUN,STEPS,SELECTED,HistoricalDirection,protected_callback,training_rows,exposure,space,state_hash
from run_rmu_adaptation import configuration
from run_rmu_formal import Telemetry,fingerprints,now
from rmu_validation import HeldOutValidation
from model import get_model
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
from trainer import load_trainer
from trainer.utils import seed_everything

def fixed_configuration():
 cfg=configuration(12,2,5e-5,32,RUN.name,120)
 cfg.paths.output_dir=str(RUN);cfg.trainer.args.output_dir=str(RUN);cfg.trainer.args.logging_dir=str(RUN/'logs')
 cfg.validation.selection_rule='Locked continuous trajectory:40/80/120; no score-based changes'
 actual=OmegaConf.to_container(cfg,resolve=True);old=json.loads((HISTORY/'training_config.json').read_text())
 def normalize(x):
  x=copy.deepcopy(x);x.pop('task_name');x['paths'].pop('output_dir');x['trainer']['args'].pop('output_dir');x['trainer']['args'].pop('logging_dir');x['trainer']['args'].pop('max_steps');x['validation'].pop('selection_rule');return x
 assert normalize(actual)==normalize(old),'Unexpected historical configuration change'
 assert actual['trainer']['handler']=='RMU' and actual['trainer']['method_args']['retain_loss_type']=='EMBED_DIFF'
 return cfg,actual

def preflight():
 cfg,resolved=fixed_configuration();sizes=[sum(x['bytes'] for x in json.loads(p.read_text())['parameters']) for p in HISTORY.glob('checkpoint-*/checkpoint.json')]
 # All three LFS objects persist; worktree holds small pointers. Allow one extra
 # compression/copy transient, metadata/dev/full results, and generous headroom.
 estimate={'three_delta_objects':3*max(sizes),'compression_and_lfs_temporary_duplicates':2*max(sizes),'text_results_and_logs_budget':512*2**20,'additional_margin':512*2**20}
 required=sum(estimate.values());space(required)
 manifest=json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())
 for row in manifest['files']+manifest['tokenizer_files']:
  assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 env={p:importlib.metadata.version(p) for p in ['torch','transformers','accelerate','datasets','lm_eval','deepspeed']}
 result={'recorded_at':now(),'configuration_equals_history_except_length_and_metadata':True,'historical_delta_sizes':sizes,'additional_space_estimate':estimate,'required_plus_1GiB_reserve':required+2**30,'space':shutil.disk_usage(ROOT)._asdict(),'versions':env,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'max_optimizer_updates':120,'checkpoints':list(STEPS),'expected_steps_per_epoch':128,'epoch_crossing':False,'schedule':'constant, no warmup; max_steps does not alter first-epoch sampling','lfs_quota':'unknown; do not upload new objects without verified available quota','commands':{'training':'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 envs/unlearning/bin/python -u scripts/run_rmu_step_comparison.py --run','evaluation':'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 envs/unlearning/bin/python -u scripts/evaluate_rmu_step_comparison.py'}}
 dump(ROOT/'configs/rmu_steps/continuous120.json',resolved);dump(ROOT/'logs/rmu_step_comparison_preflight.json',result)
 return cfg,result

class SafeValidation(HeldOutValidation):
 def __init__(self,*args,integrity,**kwargs):super().__init__(*args,**kwargs);self.integrity=integrity
 def evaluate(self,step,epoch):
  # Historical initial validation consumed exactly one CPU random control draw.
  # TokenBlockDataset/collator and eval forward consume no RNG before that draw.
  # Keep this one-time draw outside the generic restoration context.
  if self.trainer.control_vec is None:
   assert step==0
   self.trainer.get_control_vector(self.trainer.model.config.hidden_size)
  with protected_callback(self.trainer,self.integrity,'article_validation',step):super().evaluate(step,epoch)
  dump(self.output/'callback_integrity.json',self.integrity)

class SafetyTiming(TrainerCallback):
 def __init__(self,trainer,out):self.trainer=trainer;self.out=out;self.rows=[]
 def on_step_begin(self,args,state,control,**kwargs):space();torch.cuda.synchronize();self.start=time.perf_counter()
 def on_pre_optimizer_step(self,args,state,control,**kwargs):
  for n,p in self.trainer.model.named_parameters():
   if n in SELECTED:
    assert p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,n
   else:assert not p.requires_grad and p.grad is None,n
  assert not self.trainer.ref_model.training and all(not p.requires_grad and p.grad is None for p in self.trainer.ref_model.parameters())
 def on_step_end(self,args,state,control,**kwargs):
  torch.cuda.synchronize();self.rows.append({'step':state.global_step,'optimizer_update_loop_seconds':time.perf_counter()-self.start,'free_disk_bytes':space(),'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'lr':[g['lr'] for g in self.trainer.optimizer.param_groups]})
  assert self.rows[-1]['lr']==[5e-5]*len(self.trainer.optimizer.param_groups)
  dump(self.out/'update_timings.json',self.rows)

class Checkpoints(TrainerCallback):
 def __init__(self,trainer,tok,original,out,integrity,configuration):self.trainer=trainer;self.tok=tok;self.original=original;self.out=out;self.integrity=integrity;self.configuration=configuration;self.done=[]
 def on_log(self,args,state,control,logs=None,**kwargs):
  step=state.global_step
  if step not in STEPS or step in self.done or 'loss' not in (logs or {}):return
  rows=training_rows(self.out/'training_metrics.jsonl');assert [r['step'] for r in rows]==list(range(1,step+1))
  for row in rows:
   assert all(math.isfinite(row[k]) for k in ['forget_loss','retain_loss','total_loss','gradient_norm'])
  exp=exposure(rows,self.out,step);path=self.out/f'checkpoint-{step}'
  with protected_callback(self.trainer,self.integrity,'save_rebuild_and_development',step):
   space(300*2**20);manifest=save_delta(self.trainer.model,self.original,path)
   expected={n:tensor_hash(p) for n,p in self.trainer.model.named_parameters() if n in SELECTED}
   load_delta(self.trainer.model,path,self.original)
   assert expected=={n:tensor_hash(p) for n,p in self.trainer.model.named_parameters() if n in SELECTED}
   dump(path/'rebuild_verification.json',{'compressed_payload_and_base_and_reconstructed_hashes_match':True,'trained_parameter_hashes':expected,'bytes':sum(r['bytes'] for r in manifest['parameters']),'free_disk_bytes':space()})
   score(self.trainer.model,self.tok,path/'development')
   if step==40:self.reproduction(rows,exp,path,manifest)
  self.done.append(step);dump(self.out/'callback_integrity.json',self.integrity);dump(self.out/'completed_checkpoints.json',self.done)
 def reproduction(self,rows,exp,path,manifest):
  oldrows=training_rows(HISTORY/'training_metrics.jsonl');oldindices=json.loads((HISTORY/'consumed_block_indices.json').read_text())
  assert {r:x['indices'] for r,x in exp['roles'].items()}==oldindices,'Unexplained sampling drift: stop before update41'
  assert exp['input_tokens']==sum(r['input_tokens'] for r in oldrows)==320619
  assert exp['loss_tokens']==sum(r['loss_tokens'] for r in oldrows)==319339
  oldmanifest=json.loads((HISTORY/'checkpoint-40/checkpoint.json').read_text());oldhash={r['name']:r['trained_tensor_sha256'] for r in oldmanifest['parameters']};newhash={r['name']:r['trained_tensor_sha256'] for r in manifest['parameters']}
  keys=['forget_loss','retain_loss','total_loss','gradient_norm','learning_rate','input_tokens','loss_tokens','sampled_forget_indices','sampled_retain_indices']
  differences=[{'step':a['step'],'field':k,'old':a[k],'new':b[k]} for a,b in zip(oldrows,rows) for k in keys if a[k]!=b[k]]
  old=json.loads((HISTORY/'checkpoint-40/development/samples.json').read_text());new=json.loads((path/'development/samples.json').read_text());pred_differences={}
  for task in old:
   assert [(r['task'],r['doc_id'],r['doc_hash'],r['prompt_hash'],r['target']) for r in old[task]]==[(r['task'],r['doc_id'],r['doc_hash'],r['prompt_hash'],r['target']) for r in new[task]]
   pred_differences[task]=sum(a['prediction']!=b['prediction'] for a,b in zip(old[task],new[task]))
  exact=newhash==oldhash
  evidence={'bitwise_parameter_match':exact,'sampling_token_budget_and_direction_match':True,'configuration_match_except_length_and_metadata':True,'loss_record_differences':differences,'development_prediction_differences':pred_differences,'development_all_scores_bitwise_equal':old==new,'historical_parameter_hashes':oldhash,'new_parameter_hashes':newhash,'historical_full_source':str((REPO/'saves/unlearn/rmu_scale_v2/full_evaluations/h12_a2_s32_fixed40_checkpoint-40').relative_to(ROOT)),'reuse_historical_full':exact,'runtime_settings':{'torch_version':torch.__version__,'cuda':torch.version.cuda,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32}}
  dump(self.out/'reproduction40.json',evidence)
  if not exact or differences or any(pred_differences.values()):
   # Hold the same live optimizer/RNG state while the operator investigates.
   # No next optimizer update occurs until a written evidence-based decision.
   dump(self.out/'reproduction40_waiting.json',{'status':'paused_for_audit','weights_exact':exact,'optimizer_and_scheduler_retained_in_memory':True,'decision_path':'reproduction40_decision.json'})
   print('STEP40_AUDIT_REQUIRED: live optimizer retained; waiting for evidence-based decision',flush=True)
   decision_path=self.out/'reproduction40_decision.json'
   while not decision_path.exists():space();time.sleep(10)
   decision=json.loads(decision_path.read_text())
   assert decision['action']=='continue' and decision['all_key_conditions_verified'] and decision['explanation'] and decision['evidence'],decision
   evidence['audited_decision']=decision;dump(self.out/'reproduction40.json',evidence)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args();cfg,pf=preflight()
 if not a.run:print(json.dumps(pf,indent=2));return
 assert not RUN.exists(),'Refuse to overwrite any run';RUN.mkdir(parents=True)
 config=OmegaConf.to_container(cfg,resolve=True);dump(RUN/'training_config.json',config)
 sources=['scripts/'+n for n in ['run_rmu_step_comparison.py','rmu_step_common.py','rmu_adaptation_utils.py','rmu_validation.py','run_rmu_formal.py','run_rmu_adaptation.py']]+['open-unlearning/src/trainer/unlearn/rmu.py']
 run={'status':'starting','started_at':now(),'code_commit':pf['code_commit'],'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},'command':pf['commands']['training'],'versions':pf['versions'],'space_before':shutil.disk_usage(ROOT)._asdict(),'full_optimizer_state_persisted':False,'resume_warning':'Deltas alone cannot resume the exact optimizer trajectory','plan':'docs/rmu-step-comparison-plan.md'}
 dump(RUN/'run.json',run);start=time.perf_counter()
 try:
  seed_everything(42);model,tok=get_model(cfg.model);data=get_data(cfg.data,mode=cfg.mode,tokenizer=tok,template_args=cfg.model.template_args);collator=get_collators(cfg.collator,tokenizer=tok)
  trainer,args=load_trainer(trainer_cfg=cfg.trainer,model=model,train_dataset=data['train'],processing_class=tok,data_collator=collator,evaluators=None,template_args=cfg.model.template_args)
  trainer.create_optimizer();model.to(trainer.accelerator.device)
  selected=[n for n,p in model.named_parameters() if p.requires_grad];assert selected==SELECTED
  assert sum(p.numel() for p in model.parameters() if p.requires_grad)==74711040
  baseline=json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text());assert fingerprints(model)==baseline
  original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected}
  trainer.get_control_vector=HistoricalDirection(trainer,RUN)
  integrity=[];timing=SafetyTiming(trainer,RUN);trainer.add_callback(Telemetry(trainer,RUN/'training_metrics.jsonl',phase='train'));trainer.add_callback(timing)
  val={r:TokenBlockDataset(hf_args={'path':'json','data_files':str(cfg.validation[f'{r}_file']),'split':'train'},template_args={},tokenizer=tok,max_length=256) for r in ['forget','retain']}
  monitor=SafeValidation(trainer,val['forget'],val['retain'],collator,RUN,every_steps=10,integrity=integrity);trainer.add_callback(monitor)
  checks=Checkpoints(trainer,tok,original,RUN,integrity,config);trainer.add_callback(checks)
  assert len(data['train'])==2048 and math.ceil(len(data['train'])/16)==128
  dump(RUN/'actual_training_arguments.json',args.to_dict());run['status']='training';dump(RUN/'run.json',run)
  trainstart=time.perf_counter();result=trainer.train();trainelapsed=time.perf_counter()-trainstart
  assert trainer.state.global_step==120 and checks.done==list(STEPS)
  after=fingerprints(model);assert {n for n in baseline if baseline[n]!=after[n]}==set(SELECTED);assert fingerprints(trainer.ref_model)==baseline
  trainer.save_state();rows=training_rows(RUN/'training_metrics.jsonl');exposure(rows,RUN,120)
  run.update(status='completed',completed_at=now(),total_process_seconds=time.perf_counter()-start,trainer_train_wall_seconds_including_callbacks=trainelapsed,optimizer_update_loop_seconds=sum(r['optimizer_update_loop_seconds'] for r in timing.rows),callback_integrity=integrity,trainer_metrics=result.metrics,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),reference_and_frozen_parameters_unchanged=True,space_after=shutil.disk_usage(ROOT)._asdict())
  dump(RUN/'run.json',run);print('CONTINUOUS120_COMPLETE',flush=True)
 except BaseException as exc:
  run.update(status='failed',failed_at=now(),error_type=type(exc).__name__,error=str(exc),elapsed_seconds=time.perf_counter()-start);dump(RUN/'run.json',run);raise
if __name__=='__main__':main()
