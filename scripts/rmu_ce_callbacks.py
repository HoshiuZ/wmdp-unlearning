"""Shared logging, validation and exact checkpoint callbacks for RMU+CE."""
import json,math,time
from pathlib import Path
from datetime import datetime,timezone,timedelta
import torch
from transformers import TrainerCallback
from rmu_ce_utils import ROOT,dump,tensor_hash,save_delta,load_delta,score,preserve_rng

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def fingerprints(model):return {n:tensor_hash(p) for n,p in model.named_parameters()}

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

class HeldOutValidation(TrainerCallback):
    def __init__(self,trainer,forget,retain,collator,output_dir,every_steps=20):
        self.trainer=trainer;self.forget=forget;self.retain=retain;self.collator=collator
        self.output=Path(output_dir);self.every=every_steps;self.results=[]
        if len(forget)!=len(retain):raise ValueError('Validation roles must have equal block counts')
    def on_train_begin(self,args,state,control,**kwargs):self.evaluate(state.global_step,state.epoch)
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step%self.every==0:self.evaluate(state.global_step,state.epoch)
    def evaluate(self,step,epoch):
        model=self.trainer.model;was_training=model.training;model.eval();start=time.perf_counter()
        records=[];nll_sum=0.;nll_tokens=0
        try:
            with torch.no_grad(),self.trainer.compute_loss_context_manager():
                for i in range(len(self.forget)):
                    batch=self.collator([{'forget':self.forget[i],'retain':self.retain[i]}])
                    batch=self.trainer._prepare_inputs(batch)
                    self.trainer.compute_loss(model,batch)
                    records.append(dict(self.trainer.last_loss_record))
                    retain={k:v for k,v in batch['retain'].items() if k in ('input_ids','attention_mask','labels')}
                    loss=model(**retain).loss
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite held-out retain NLL')
                    count=int((retain['labels'][:,1:]!=-100).sum())
                    nll_sum+=float(loss)*count;nll_tokens+=count
        finally:model.train(was_training)
        nll=nll_sum/nll_tokens
        elapsed=time.perf_counter()-start
        cuda=next(model.parameters()).is_cuda
        input_tokens=sum(r['input_tokens'] for r in records)
        row={'training_run_id':self.output.name,'phase':'validation','step':step,'epoch':epoch,
             'forget_loss':sum(r['forget_loss'] for r in records)/len(records),
             'retain_loss':sum(r['retain_loss'] for r in records)/len(records),
             'total_loss':sum(r['total_loss'] for r in records)/len(records),
             'validation_loss':sum(r['total_loss'] for r in records)/len(records),
             'retain_nll':nll,'retain_ppl':math.exp(nll) if nll<700 else None,
             'nll_tokens':nll_tokens,'blocks_per_role':len(records),
             'learning_rate':None,'gradient_norm':None,
             'elapsed_seconds':elapsed,'recorded_at':now(),
             'input_tokens':input_tokens,'tokens_per_second':input_tokens/elapsed,
             'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if cuda else None,
             'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if cuda else None,
             'checkpoint_path':None,'sampling':'fixed held-out blocks; same control vector as training'}
        self.output.mkdir(parents=True,exist_ok=True)
        for name in ('validation_metrics.jsonl','training_metrics.jsonl'):
            with (self.output/name).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.results.append(row)
        print(f"Validation step {step}: forget={row['forget_loss']:.6f}, retain={row['retain_loss']:.6f}, retain_nll={nll:.6f}",flush=True)
