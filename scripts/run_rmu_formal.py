"""Run the prepared 80-step formal RMU training and verify save/reload.

Uses the OpenUnlearning model/data/trainer factories. Does not run WMDP/MMLU.
"""
import sys
import os
import json
import math
import time
import hashlib
import subprocess
from datetime import datetime,timezone,timedelta
from pathlib import Path
import torch
from transformers import TrainerCallback, AutoModelForCausalLM, AutoTokenizer
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'open-unlearning'
sys.path.insert(0,str(REPO/'src'));os.chdir(REPO)
from model import get_model
from data import get_data,get_collators
from trainer import load_trainer
from trainer.utils import seed_everything
from data.pretraining import TokenBlockDataset
from rmu_validation import HeldOutValidation


def now(): return datetime.now(timezone(timedelta(hours=8))).isoformat()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def fingerprints(model): return {n:tensor_hash(p) for n,p in model.named_parameters()}


class Telemetry(TrainerCallback):
    def __init__(self,trainer,path,phase="smoke_train"):
        self.phase=phase
        self.trainer=trainer;self.path=path;self.start=None;self.last=None;self.total_tokens=0
    def on_train_begin(self,args,state,control,**kwargs):
        self.start=self.last=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def on_log(self,args,state,control,logs=None,**kwargs):
        if not self.trainer.step_loss_records:return
        records=self.trainer.step_loss_records;self.trainer.step_loss_records=[]
        current=time.perf_counter();elapsed=current-self.last
        tokens=sum(r['input_tokens'] for r in records);self.total_tokens+=tokens
        row={'training_run_id':Path(args.output_dir).name,'phase':self.phase,
             'step':state.global_step,'epoch':state.epoch,
             'forget_loss':sum(r['forget_loss'] for r in records)/len(records),
             'retain_loss':sum(r['retain_loss'] for r in records)/len(records),
             'total_loss':sum(r['total_loss'] for r in records)/len(records),
             'validation_loss':None,'learning_rate':logs.get('learning_rate'),
             'gradient_norm':logs.get('grad_norm'),'input_tokens':tokens,
             'loss_tokens':sum(r['loss_tokens'] for r in records),'total_input_tokens':self.total_tokens,
             'tokens_per_second':tokens/elapsed,'elapsed_seconds':current-self.start,
             'interval_seconds':elapsed,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
             'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
             'checkpoint_path':None,'recorded_at':now(),
             'sampled_forget_indices':[i for r in records for i in (r['sampled_indices']['forget'] or [])],
             'sampled_retain_indices':[i for r in records for i in (r['sampled_indices']['retain'] or [])]}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.last=current


def main():
    with initialize_config_dir(config_dir=str(REPO/'configs'),version_base=None):
        cfg=compose(config_name='train',overrides=['experiment=unlearn/wmdp/qwen3_4b_bio_formal'])
    cfg.paths.work_dir=str(REPO)
    output=(REPO/str(cfg.paths.output_dir)).resolve()
    if output.exists():raise RuntimeError(f'Refuse to overwrite existing run: {output}')
    output.mkdir(parents=True)
    resolved=OmegaConf.to_container(cfg,resolve=True)
    dump(output/'training_config.json',resolved)
    run={'run_id':cfg.task_name,'started_at':now(),'status':'running',
         'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'command':'envs/unlearning/bin/python -u scripts/run_rmu_formal.py',
         'base_model':str(cfg.model.model_args.pretrained_model_name_or_path),
         'subset_manifest':'logs/rmu_formal_blocks_manifest.json','max_steps':80}
    dump(ROOT/'logs/rmu_formal_run.json',run)
    manifest=json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())
    for info in manifest['files']:
        assert hashlib.sha256((ROOT/info['path']).read_bytes()).hexdigest()==info['sha256']
    for info in manifest['tokenizer_files']:
        assert hashlib.sha256((ROOT/info['path']).read_bytes()).hexdigest()==info['sha256']
    seed_everything(cfg.trainer.args.seed)
    print('Loading official starting weights',flush=True)
    model,tokenizer=get_model(cfg.model)
    data=get_data(cfg.data,mode=cfg.mode,tokenizer=tokenizer,template_args=cfg.model.template_args)
    collator=get_collators(cfg.collator,tokenizer=tokenizer)
    trainer,args=load_trainer(trainer_cfg=cfg.trainer,model=model,train_dataset=data['train'],
                             processing_class=tokenizer,data_collator=collator,
                             evaluators=None,template_args=cfg.model.template_args)
    trainer.create_optimizer()
    selected=[n for n,p in model.named_parameters() if p.requires_grad]
    assert len(selected)==3 and not any(p.requires_grad for p in trainer.ref_model.parameters())
    print('Taking weight fingerprints; selected parameters:',selected,flush=True)
    before=fingerprints(model)
    baseline=json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text())
    assert before==baseline, 'Starting weights differ from the verified original baseline'
    original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in selected}
    dump(output/'base_weight_fingerprints.json',before)
    trainer.add_callback(Telemetry(trainer,output/'training_metrics.jsonl',phase='train'))
    validation_sets={}
    for role in ('forget','retain'):
        validation_sets[role]=TokenBlockDataset(
            hf_args={'path':'json','data_files':str(cfg.validation[f'{role}_file']),'split':'train'},
            template_args=cfg.model.template_args,tokenizer=tokenizer,max_length=256)
    monitor=HeldOutValidation(trainer,validation_sets['forget'],validation_sets['retain'],collator,
                              output,every_steps=cfg.validation.every_steps)
    trainer.add_callback(monitor)
    print('Beginning eighty optimizer steps',flush=True)
    result=trainer.train()
    assert trainer.state.global_step==80
    trainer.save_state()
    after=fingerprints(model)
    changed=[n for n in before if before[n]!=after[n]]
    assert changed and set(changed).issubset(set(selected)),changed
    changes={}
    for n,p in model.named_parameters():
        if n in original:
            diff=p.detach().cpu().float()-original[n].float()
            changes[n]={'changed_elements':int((diff!=0).sum()),'max_abs_change':float(diff.abs().max()),
                        'mean_abs_change':float(diff.abs().mean())}
    peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
    model_dir=output/'model'
    print('Saving final formal weights in <=1GB shards',flush=True)
    model.save_pretrained(model_dir,max_shard_size='1GB',safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    if trainer.control_vec is not None:
        torch.save(trainer.control_vec.detach().cpu(),output/'control_vector.pt')
    dump(output/'trained_weight_fingerprints.json',after)
    # Free both GPU copies before reloading the saved checkpoint.
    validation_results=monitor.results
    del data,collator,original,trainer,model,monitor,validation_sets
    import gc
    gc.collect();torch.cuda.empty_cache()
    print('Reloading saved checkpoint',flush=True)
    reloaded=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True,device_map='cuda')
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    reload_hashes=fingerprints(reloaded)
    assert after==reload_hashes
    encoded=tok('A short compatibility check.',return_tensors='pt').to('cuda')
    reloaded.eval()
    with torch.no_grad(): logits=reloaded(**encoded).logits
    assert torch.isfinite(logits).all()
    rows=[json.loads(line) for line in (output/'training_metrics.jsonl').read_text().splitlines()]
    train_rows=[r for r in rows if r['phase']=='train']
    assert [r['step'] for r in train_rows]==list(range(1,81))
    assert [r['step'] for r in validation_results]==[0,20,40,60,80]
    consumed={role:[i for r in train_rows for i in r[f'sampled_{role}_indices']] for role in ('forget','retain')}
    assert all(len(indices)==1280 for indices in consumed.values())
    dump(output/'consumed_block_indices.json',consumed)
    actual_used_counts={role:len(set(indices)) for role,indices in consumed.items()}
    assert all(math.isfinite(r[k]) for r in rows for k in ['forget_loss','retain_loss','total_loss'])
    artifacts=[]
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            artifacts.append({'path':str(path.relative_to(ROOT)),'size_bytes':path.stat().st_size,'sha256':h.hexdigest()})
    run.update(status='completed',completed_at=now(),global_step=80,validation_results=validation_results,actual_used_block_counts=actual_used_counts,
               training_metrics=result.metrics,selected_parameters=selected,changed_parameters=changed,
               parameter_changes=changes,frozen_parameters_unchanged=True,
               cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved,
               reload_fingerprints_identical=True,reload_forward_finite=True,
               training_metrics_path=str((output/'training_metrics.jsonl').relative_to(ROOT)),
               model_directory=str(model_dir.relative_to(ROOT)),model_files=artifacts,
               note='Formal RMU training; fixed final step80 checkpoint. Official WMDP/MMLU post-training evaluation remains pending.')
    dump(ROOT/'logs/rmu_formal_run.json',run);dump(output/'training_validation.json',run)
    print(json.dumps({k:run[k] for k in ['status','global_step','changed_parameters','cuda_peak_allocated_bytes','reload_fingerprints_identical','reload_forward_finite']},indent=2),flush=True)

if __name__=='__main__': main()
