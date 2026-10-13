"""Controlled RMU/SimNPO ablations, shared masks and explicit loss coefficients."""
import torch
import torch.nn.functional as F
from trainer.unlearn.rmu import RMU

def token_nll(logits,labels):
 target=labels[:,1:];valid=target!=-100;counts=valid.sum(-1)
 if (counts==0).any():raise ValueError('No effective prediction tokens')
 values=F.cross_entropy(logits[:,:-1].float().transpose(1,2),target,reduction='none',ignore_index=-100)
 return values.sum(-1)/counts

def capped_weight(rmu_norm,sim_norm,upper,rho=.5,eps=1e-12):
 import math
 if not all(math.isfinite(x) and x>=0 for x in [rmu_norm,sim_norm,upper]):raise FloatingPointError('Invalid controller norm')
 if rmu_norm<=eps:return 0.
 if sim_norm<=eps:return upper
 return min(upper,rho*rmu_norm/(sim_norm+eps))

def simnpo(logits,labels,beta=.1,delta=0.):
 return -2/beta*F.logsigmoid(beta*(token_nll(logits,labels)-delta)).mean()

class ControlledSimRMU(RMU):
 def __init__(self,variant='joint',sim_weight=1.,ce_weight=1.,beta=.1,gradient_cap=False,*a,**kw):
  super().__init__(*a,**kw);self.variant=variant;self.sim_weight=sim_weight;self.ce_weight=ce_weight;self.beta=beta;self.component_records=[];self.gradient_cap=gradient_cap;self.last_sim_weight=sim_weight;self.controller_records=[]
 def components(self,model,inputs):
  clean={r:{k:inputs[r][k] for k in ['input_ids','attention_mask','labels']} for r in ['forget','retain']}
  f=clean['forget'];r=clean['retain']
  fact,fo=self.forward_with_cache(model,f,self.model_module,no_grad=False)
  target=self.get_control_vector(fact.shape[-1]).to(fact).expand_as(fact)
  fm=(f['labels']!=-100)&f['attention_mask'].bool()
  result={'rmu_forget':self.compute_activation_loss(fact,target,fm),'sim_forget':simnpo(fo.logits,f['labels'],self.beta)}
  ract,ro=self.forward_with_cache(model,r,self.model_module,no_grad=False)
  result['retain_ce']=token_nll(ro.logits,r['labels']).mean()
  if self.variant!='sim':
   ref,_=self.forward_with_cache(self.ref_model,r,self.ref_module,no_grad=True)
   result['retain_repr']=self.compute_activation_loss(ract,ref.to(ract),(r['labels']!=-100)&r['attention_mask'].bool())
  else:result['retain_repr']=result['retain_ce'].new_zeros(())
  return result,fo,clean
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  c,fo,clean=self.components(model,inputs)
  weight=self.sim_weight
  if self.gradient_cap:
   if model.training and torch.is_grad_enabled():
    params=[p for p in model.parameters() if p.requires_grad];norms={}
    for key in ['rmu_forget','sim_forget']:
     gradients=torch.autograd.grad(c[key],params,retain_graph=True);norms[key]=float(torch.stack([g.float().square().sum() for g in gradients]).sum().sqrt());del gradients
    weight=capped_weight(norms['rmu_forget'],norms['sim_forget'],self.sim_weight);self.last_sim_weight=weight
    ratio=weight*norms['sim_forget']/norms['rmu_forget'] if norms['rmu_forget']>1e-12 else None
    assert ratio is None or ratio<=.500001
    self.controller_records.append({'step':self.state.global_step,'rmu_gradient_norm':norms['rmu_forget'],'sim_gradient_norm':norms['sim_forget'],'effective_sim_weight':weight,'raw_weighted_ratio':self.sim_weight*norms['sim_forget']/norms['rmu_forget'] if norms['rmu_forget']>1e-12 else None,'capped_ratio':ratio,'cap_active':weight<self.sim_weight})
   else:weight=self.last_sim_weight
  f=weight*c['sim_forget'] if self.variant=='sim' else c['rmu_forget']
  if self.variant=='joint':f=f+weight*c['sim_forget']
  r=self.ce_weight*c['retain_ce']+(self.alpha*c['retain_repr'] if self.variant!='sim' else 0)
  loss=f+r
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined loss')
  self.last_loss_record={'forget_loss':float(f.detach()),'retain_loss':float(r.detach()),'total_loss':float(loss.detach()),'input_tokens':sum(int(clean[k]['attention_mask'].sum()) for k in clean),'loss_tokens':sum(int((clean[k]['labels']!=-100).sum()) for k in clean),'sampled_indices':{k:inputs[k]['index'].detach().cpu().tolist() for k in clean}}
  if model.training:
   self.step_loss_records.append(self.last_loss_record)
   self.component_records.append({'step':self.state.global_step,'effective_prediction_tokens':sum(int((clean[k]['labels'][:,1:]!=-100).sum()) for k in clean),'components':{k:float(v.detach()) for k,v in c.items()}})
  return (loss,fo) if return_outputs else loss
