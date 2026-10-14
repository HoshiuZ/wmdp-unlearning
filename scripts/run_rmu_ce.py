"""Reproduce the retained RMU+CE configuration; never overwrite retained runs."""
import os,json,time,argparse,subprocess,shutil,hashlib
from pathlib import Path
import torch
from omegaconf import OmegaConf
from rmu_ce_utils import ROOT,REPO,BASE,OUT,dump,tensor_hash
from rmu_ce_callbacks import Telemetry,Checkpoints,HeldOutValidation,fingerprints,now
from rmu_ce_method import RMUCE
from model import get_model
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
from trainer import TRAINER_REGISTRY,load_trainer
from trainer.utils import seed_everything
TRAINER_REGISTRY['RMUCE']=RMUCE

def build(seed,out):
 os.chdir(REPO);cfg=OmegaConf.create(json.loads((ROOT/f'configs/rmu_ce/seed{seed}.json').read_text()));cfg.model.model_args.pretrained_model_name_or_path=str(BASE);cfg.model.tokenizer_args.pretrained_model_name_or_path=str(BASE);cfg.paths.output_dir=str(out);cfg.paths.work_dir=str(REPO);cfg.trainer.args.output_dir=str(out);cfg.trainer.args.logging_dir=str(out/'logs')
 seed_everything(seed);model,tok=get_model(cfg.model);data=get_data(cfg.data,mode=cfg.mode,tokenizer=tok,template_args=cfg.model.template_args);collator=get_collators(cfg.collator,tokenizer=tok);trainer,_=load_trainer(cfg.trainer,model,train_dataset=data['train'],processing_class=tok,data_collator=collator,evaluators=None,template_args=cfg.model.template_args);trainer.create_optimizer();model.to(trainer.accelerator.device)
 selected=[n for n,p in model.named_parameters() if p.requires_grad];assert selected==[f'model.layers.{i}.mlp.down_proj.weight' for i in [10,11,12]];before=fingerprints(model);assert before==json.loads((ROOT/'reference/rmu_ce/base_weight_fingerprints.json').read_text());assert not trainer.ref_model.training and not any(p.requires_grad for p in trainer.ref_model.parameters())
 saved=torch.tensor(json.loads((ROOT/f'reference/rmu_ce/seed{seed}/control_vector.json').read_text())['values']).reshape(1,1,-1);generator=trainer.get_control_vector
 def fixed_vector(dim):
  generator(dim);trainer.control_vec=saved;return trainer.control_vec
 trainer.get_control_vector=fixed_vector
 return cfg,trainer,tok,data,collator,before

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args();out=OUT/f'reproductions/seed{a.seed}_'+('minimal' if a.check_only else 'train40');assert not out.exists();assert shutil.disk_usage(ROOT).free>3*2**30;out.mkdir(parents=True)
 for q in json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())['files']:assert hashlib.sha256((ROOT/q['path']).read_bytes()).hexdigest()==q['sha256']
 cfg,t,tok,data,col,before=build(a.seed,out);model=t.model;selected=[n for n,p in model.named_parameters() if p.requires_grad];original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected};start=time.perf_counter();torch.cuda.reset_peak_memory_stats();run={'seed':a.seed,'status':'running','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'max_steps':2 if a.check_only else 40,'started_at':now()};dump(out/'run.json',run);dump(out/'training_config.json',OmegaConf.to_container(cfg,resolve=True))
 if a.check_only:
  model.train();checks=[]
  for i in range(2):
   batch=t._prepare_inputs(col([data['train'][i]]));t.optimizer.zero_grad();loss=t.compute_loss(model,batch);loss.backward();grads={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.requires_grad};assert all(v>0 and __import__('math').isfinite(v) for v in grads.values());assert all(p.grad is None for n,p in model.named_parameters() if n not in selected);assert all(p.grad is None for p in t.ref_model.parameters());torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);t.optimizer.step();checks.append({'step':i+1,'loss':float(loss.detach()),'gradients':grads})
  run['checks']=checks
 else:
  val={r:TokenBlockDataset(hf_args={'path':'json','data_files':str(cfg.validation[r+'_file']),'split':'train'},template_args={},tokenizer=tok,max_length=256) for r in ['forget','retain']};monitor=HeldOutValidation(t,val['forget'],val['retain'],col,out,every_steps=10);cp=Checkpoints(t,tok,original,out);t.add_callback(Telemetry(t,out/'training_metrics.jsonl',phase='train'));t.add_callback(monitor);t.add_callback(cp);result=t.train();assert t.state.global_step==40;t.save_state();rows=[json.loads(l) for l in (out/'training_metrics.jsonl').open()];train=[r for r in rows if r['phase']=='train'];consumed={r:[i for row in train for i in row['sampled_'+r+'_indices']] for r in ['forget','retain']};assert consumed==json.loads((ROOT/f'reference/rmu_ce/seed{a.seed}/consumed_block_indices.json').read_text());budget=json.loads((ROOT/f'reference/rmu_ce/seed{a.seed}/budget.json').read_text());assert sum(r['input_tokens'] for r in train)==budget['actual_input_tokens'];assert sum(r['loss_tokens'] for r in train)==budget['actual_loss_tokens'];dump(out/'consumed_block_indices.json',consumed);dump(out/'component_metrics.json',t.component_records);run.update(training_metrics=result.metrics,development=cp.rows,validation=monitor.results,budget=budget)
 after=fingerprints(model);assert {n for n in before if before[n]!=after[n]}==set(selected);assert fingerprints(t.ref_model)==before;run.update(status='completed',seconds=time.perf_counter()-start,frozen_reference_unchanged=True,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved());dump(out/'run.json',run);print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
