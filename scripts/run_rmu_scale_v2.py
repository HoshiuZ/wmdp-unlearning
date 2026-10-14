"""One sequential, space-gated hook12 RMU trial; first trial final delta only."""
import os,sys,json,time,argparse,subprocess,shutil
from pathlib import Path
import torch
from transformers import TrainerCallback
from omegaconf import OmegaConf
from rmu_adaptation_utils import ROOT,REPO,dump,score,save_delta,load_delta,preserve_rng,tensor_hash,OUT as LEGACY
from run_rmu_adaptation import configuration
from run_rmu_formal import Telemetry,fingerprints,now
from model import get_model
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
from trainer import load_trainer
from trainer.utils import seed_everything
from rmu_validation import HeldOutValidation
OUT=REPO/'saves/unlearn/rmu_scale_v2'
class Monitor(TrainerCallback):
 def __init__(self,trainer,tok,original,out,save_middle):self.trainer=trainer;self.tok=tok;self.original=original;self.out=out;self.save_middle=save_middle;self.rows=[]
 def on_step_end(self,args,state,control,**kwargs):
  step=state.global_step
  if step not in [10,20,40]:return
  model=self.trainer.model;was=model.training;path=self.out/f'checkpoint-{step}';saved=step==40 or self.save_middle
  with preserve_rng():
   if saved:
    assert shutil.disk_usage(ROOT).free>2**30+260*2**20,'Space reserve violated; stop without deleting prior artifacts'
    manifest=save_delta(model,self.original,path);load_delta(model,path,self.original)
   metrics=score(model,self.tok,path/'development')
  model.train(was);self.rows.append({'step':step,'metrics':metrics,'path':str(path.relative_to(ROOT)),'weights_saved':saved,'checkpoint_bytes':sum(r['bytes'] for r in manifest['parameters']) if saved else 0,'save_reload_identical':True if saved else None})
  dump(self.out/'development_results.json',self.rows);print('DEVELOPMENT',step,metrics,'saved',saved,flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--steering',type=float,required=True);p.add_argument('--alpha',type=float,default=1);p.add_argument('--save-intermediate',action='store_true');p.add_argument('--coverage-dir');a=p.parse_args()
 name=f'h12_a{a.alpha:g}_s{a.steering:g}_'+('coverage40' if a.coverage_dir else 'fixed40');out=OUT/name
 assert not out.exists(),'Never overwrite trial';assert shutil.disk_usage(ROOT).free>2**30+(580 if a.save_intermediate else 260)*2**20
 out.mkdir(parents=True);cfg=configuration(12,a.alpha,5e-5,a.steering,name,40);cfg.paths.output_dir=str(out);cfg.trainer.args.output_dir=str(out);cfg.trainer.args.logging_dir=str(out/'logs');cfg.validation.selection_rule='weighted MMLU Pareto; separate 2pp/3pp references; exploratory'
 resolved=OmegaConf.to_container(cfg,resolve=True)
 if a.coverage_dir:
  for role in ['forget','retain']:resolved['data'][role][f'WMDP_{role}']['handler']='PositionBlockDataset';resolved['data'][role][f'WMDP_{role}']['args']['hf_args']['data_files']=str(Path(a.coverage_dir)/role/'train.jsonl')
 dump(out/'training_config.json',resolved)
 run={'name':name,'started_at':now(),'status':'running','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'command':' '.join(sys.argv),'hook':12,'alpha':a.alpha,'steering':a.steering,'lr':5e-5,'steps':40,'seed':42,'saved_steps':[10,20,40] if a.save_intermediate else [40],'data_manifest':'logs/rmu_formal_blocks_manifest.json','coverage_dir':a.coverage_dir,'space_before':shutil.disk_usage(ROOT)._asdict()}
 dump(out/'run.json',run)
 for row in json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())['files']:assert __import__('hashlib').sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
 seed_everything(42);model,tok=get_model(cfg.model);data=get_data(cfg.data,mode=cfg.mode,tokenizer=tok,template_args=cfg.model.template_args);collator=get_collators(cfg.collator,tokenizer=tok)
 if a.coverage_dir:
  coverage_manifest=json.loads((Path(a.coverage_dir)/'manifest.json').read_text())
  for row in coverage_manifest['files']:assert __import__('hashlib').sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
  run['coverage_manifest']='logs/rmu_scale_v2_coverage_manifest.json';run['actual_data_version']=coverage_manifest['version'];dump(out/'run.json',run)
  from rmu_position_blocks import PositionBlockDataset
  for role in ['forget','retain']:setattr(data['train'],role,PositionBlockDataset(Path(a.coverage_dir)/role/'train.jsonl',role,tok))
 trainer,args=load_trainer(trainer_cfg=cfg.trainer,model=model,train_dataset=data['train'],processing_class=tok,data_collator=collator,evaluators=None,template_args=cfg.model.template_args);trainer.create_optimizer();model.to(trainer.accelerator.device)
 selected=[n for n,p in model.named_parameters() if p.requires_grad];assert selected==[f'model.layers.{i}.mlp.down_proj.weight' for i in [10,11,12]]
 before=fingerprints(model);assert before==json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text());original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected}
 stored=json.loads((LEGACY/'h12_a1_lr5e-05_s2_screen40/control_vector.json').read_text());unit=torch.tensor(stored['values']).reshape(1,1,-1)/2
 generator=trainer.get_control_vector
 def fixed_direction(dim):
  v=generator(dim);assert torch.allclose(v/a.steering,unit,atol=1e-7,rtol=0)
  trainer.control_vec=unit*a.steering;return trainer.control_vec
 trainer.get_control_vector=fixed_direction
 trainer.add_callback(Telemetry(trainer,out/'training_metrics.jsonl',phase='train'))
 val={r:TokenBlockDataset(hf_args={'path':'json','data_files':str(cfg.validation[f'{r}_file']),'split':'train'},template_args={},tokenizer=tok,max_length=256) for r in ['forget','retain']}
 monitor=HeldOutValidation(trainer,val['forget'],val['retain'],collator,out,every_steps=10);trainer.add_callback(monitor);check=Monitor(trainer,tok,original,out,a.save_intermediate);trainer.add_callback(check)
 start=time.perf_counter();result=trainer.train();assert trainer.state.global_step==40;trainer.save_state()
 after=fingerprints(model);changed=[n for n in before if before[n]!=after[n]];assert set(changed)==set(selected);assert fingerprints(trainer.ref_model)==before
 rows=[json.loads(l) for l in (out/'training_metrics.jsonl').open()];tr=[r for r in rows if r['phase']=='train'];assert [r['step'] for r in tr]==list(range(1,41));consumed={r:[i for row in tr for i in row[f'sampled_{r}_indices']] for r in ['forget','retain']}
 reference=json.loads((LEGACY/'h12_a1_lr5e-05_s2_screen40/consumed_block_indices.json').read_text());assert consumed==reference,'Article sampling order drift'
 dump(out/'consumed_block_indices.json',consumed);v=trainer.control_vec.cpu();dump(out/'control_vector.json',{'values':v.flatten().tolist(),'l2_norm':float(v.norm()),'sha256':tensor_hash(v),'unit_direction_sha256':tensor_hash(v/a.steering),'saved_original_direction_sha256':tensor_hash(unit)})
 if a.coverage_dir:
  for role in consumed:
   ds=getattr(data['train'],role);used=ds.exposures[:len(consumed[role])];assert [x['index'] for x in used]==consumed[role];dump(out/f'{role}_consumed_positions.json',used)
  origtr=[json.loads(l) for l in (LEGACY/'h12_a1_lr5e-05_s2_screen40/training_metrics.jsonl').open() if json.loads(l)['phase']=='train'];assert sum(r['input_tokens'] for r in tr)==sum(r['input_tokens'] for r in origtr),'Coverage token budget mismatch'
 run.update(status='completed',completed_at=now(),elapsed_seconds=time.perf_counter()-start,selected_parameters=selected,frozen_parameters_unchanged=True,reference_unchanged=True,sampling_sequence_identical=True,control_direction_identical=True,actual_input_tokens=sum(r['input_tokens'] for r in tr),actual_loss_tokens=sum(r['loss_tokens'] for r in tr),training_metrics=result.metrics,validation=monitor.results,development=check.rows,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),space_after=shutil.disk_usage(ROOT)._asdict());dump(out/'run.json',run);print('COMPLETE',name,flush=True)
if __name__=='__main__':main()
