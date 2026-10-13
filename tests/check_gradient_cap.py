import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'open-unlearning/src'))
import torch
from simnpo_rmu_method import capped_weight,ControlledSimRMU
assert abs(capped_weight(1,10,.15)-.05)<1e-12
assert capped_weight(1,.01,.15)==.15
assert capped_weight(0,1,.15)==0
assert capped_weight(1,0,.15)==.15
try:capped_weight(float('nan'),1,.15);raise AssertionError()
except FloatingPointError:pass
# Exercise production compute_loss, preserving preexisting accumulated gradients.
p=torch.nn.Parameter(torch.tensor([1.,2.]));p.grad=torch.tensor([7.,8.]);old=p.grad.clone()
class Fake:
 training=True
 def parameters(self):return [p]
model=Fake()
from types import SimpleNamespace
trainer=SimpleNamespace(sim_weight=.15,ce_weight=0.,alpha=0.,variant='joint',gradient_cap=True,last_sim_weight=.15,controller_records=[],component_records=[],step_loss_records=[],state=SimpleNamespace(global_step=0))
inputs={k:{'index':torch.tensor([0])} for k in ['forget','retain']};clean={k:{'attention_mask':torch.ones(1,2,dtype=torch.long),'labels':torch.tensor([[-100,1]])} for k in inputs}
trainer.components=lambda m,i:({'rmu_forget':p.square().sum(),'sim_forget':(10*p).square().sum(),'retain_ce':p.sum()*0,'retain_repr':p.sum()*0},None,clean)
loss=ControlledSimRMU.compute_loss(trainer,model,inputs);assert torch.equal(p.grad,old);loss.backward();expected=2*p.detach()+trainer.last_sim_weight*200*p.detach();assert torch.allclose(p.grad-old,expected);assert trainer.controller_records[0]['capped_ratio']<=.500001
trainer.controller_records.clear();model.training=False
with torch.no_grad():ControlledSimRMU.compute_loss(trainer,model,inputs)
assert not trainer.controller_records
print('BOUNDARY_ACCUMULATION_DETACH_EVAL_CHECKS_OK')
