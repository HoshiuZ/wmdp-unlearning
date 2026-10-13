import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'open-unlearning/src'));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from types import SimpleNamespace
from simnpo_rmu_method import simnpo,token_nll
from trainer.unlearn.simnpo import SimNPO
labels=torch.tensor([[1,2,3,-100],[2,1,-100,-100]]);logits=torch.randn(2,4,5,requires_grad=True)
class Fake:
 def __call__(self,**kw):
  assert set(kw)=={'input_ids','attention_mask','labels'}
  return SimpleNamespace(logits=logits)
inputs={'forget':{'input_ids':torch.zeros_like(labels),'attention_mask':labels!=-100,'labels':labels,'index':torch.tensor([0,1])},'retain':{'input_ids':torch.zeros_like(labels),'attention_mask':labels!=-100,'labels':labels}}
trainer=SimpleNamespace(beta=.1,delta=0.,gamma=1.,alpha=0.,compute_retain_loss=lambda **kw:torch.tensor(0.))
actual=SimNPO.compute_loss(trainer,Fake(),inputs);expect=simnpo(logits,labels);assert torch.allclose(actual,expect,atol=1e-6)
changed=logits.detach().clone();changed[0,2:]=100;changed[1,1:]=100;assert torch.allclose(simnpo(changed,labels),expect)
actual.backward();assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0
try:token_nll(logits,torch.full_like(labels,-100));raise AssertionError('zero mask accepted')
except ValueError:pass
print('MASK_SHIFT_METADATA_GRADIENT_CHECKS_OK')
