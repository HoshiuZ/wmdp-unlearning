"""Isolated CPU probe of new callback isolation and length-only configuration."""
import sys,tempfile,random
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,TrainingArguments
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'open-unlearning/src')]
from trainer.unlearn.rmu import RMU
from rmu_step_common import protected_callback,HistoricalDirection,state_hash,tensor_hash,STEPS
from run_rmu_step_comparison import fixed_configuration
cfg,_=fixed_configuration();assert cfg.trainer.args.max_steps==120 and STEPS==(40,80,120)
with tempfile.TemporaryDirectory() as directory:
 torch.manual_seed(42)
 model=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=13,num_attention_heads=2,num_key_value_heads=2,head_dim=8))
 args=TrainingArguments(output_dir=directory,use_cpu=True,report_to=[],optim='adamw_torch',learning_rate=5e-5,lr_scheduler_type='constant',max_grad_norm=1)
 trainer=RMU(model=model,args=args,alpha=2,gamma=1,retain_loss_type='EMBED_DIFF',steering_coeff=32,module_regex=r'model\.layers\.12',trainable_params_regex=[r'model\.layers\.(10|11|12)\.mlp\.down_proj\.weight'])
 trainer.create_optimizer();trainer.create_scheduler(120)
 before={n:tensor_hash(p) for n,p in model.named_parameters()};ref={n:tensor_hash(p) for n,p in trainer.ref_model.named_parameters()}
 ids=torch.randint(0,32,(1,8));labels=ids.clone();labels[:,0]=-100;labels[:,-2:]=-100
 batch={'input_ids':ids,'labels':labels,'attention_mask':(labels!=-100).long()};batch['attention_mask'][:,0]=1
 loss=trainer.compute_loss(model,{'forget':batch,'retain':batch});loss.backward()
 selected=[n for n,p in model.named_parameters() if p.requires_grad]
 assert len(selected)==3 and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
 assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
 trainer.optimizer.step();trainer.optimizer.zero_grad()
 assert {n for n,p in model.named_parameters() if tensor_hash(p)!=before[n]}==set(selected)
 assert ref=={n:tensor_hash(p) for n,p in trainer.ref_model.named_parameters()} and not trainer.ref_model.training
 trainer.step_loss_records=[];rng=state_hash((random.getstate(),np.random.get_state(),torch.get_rng_state()));checks=[]
 with protected_callback(trainer,checks,'probe',0):
  random.random();np.random.rand();torch.rand(5);model.eval()
  with torch.no_grad():trainer.compute_loss(model,{'forget':batch,'retain':batch})
 assert rng==state_hash((random.getstate(),np.random.get_state(),torch.get_rng_state())) and model.training
 # Mock a known first-draw direction, checking that it is consumed once.
 class Fake:
  control_vec=None
  def generate(self,dim):
   if self.control_vec is None:
    x=torch.rand(1,1,dim);self.control_vec=x/x.norm()*32
   return self.control_vec
 fake=Fake();torch.manual_seed(100);x=torch.rand(1,1,16);unit=x/x.norm();expected_rng=torch.get_rng_state().clone();torch.manual_seed(100)
 direction=HistoricalDirection.__new__(HistoricalDirection);direction.trainer=fake;direction.out=Path(directory);direction.generate=fake.generate;direction.unit=unit;direction.vector=unit*32
 direction(16);assert torch.equal(torch.get_rng_state(),expected_rng);direction(16);assert torch.equal(torch.get_rng_state(),expected_rng)
print('PASS: length-only config; exact selective updates; reference frozen; callback state/RNG isolation; first control draw consumed once')
