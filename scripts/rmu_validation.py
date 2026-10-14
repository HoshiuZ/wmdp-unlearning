"""Held-out representation and retain-NLL monitoring, separate from official tests."""
import json,math,time
from pathlib import Path
import torch
from transformers import TrainerCallback
from run_rmu_smoke import now

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
