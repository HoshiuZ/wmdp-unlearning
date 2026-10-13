"""Controlled RMU/SimNPO ablations, shared masks and explicit loss coefficients."""
import torch
import torch.nn.functional as F
from trainer.unlearn.rmu import RMU

def token_nll(logits,labels):
 target=labels[:,1:];valid=target!=-100;counts=valid.sum(-1)
 if (counts==0).any():raise ValueError('No effective prediction tokens')
 values=F.cross_entropy(logits[:,:-1].float().transpose(1,2),target,reduction='none',ignore_index=-100)
 return values.sum(-1)/counts

def simnpo(logits,labels,beta=.1,delta=0.):
 return -2/beta*F.logsigmoid(beta*(token_nll(logits,labels)-delta)).mean()

class ControlledSimRMU(RMU):
 def __init__(self,variant='joint',sim_weight=1.,ce_weight=1.,beta=.1,*a,**kw):
  super().__init__(*a,**kw);self.variant=variant;self.sim_weight=sim_weight;self.ce_weight=ce_weight;self.beta=beta;self.component_records=[]
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
  f=self.sim_weight*c['sim_forget'] if self.variant=='sim' else c['rmu_forget']
  if self.variant=='joint':f=f+self.sim_weight*c['sim_forget']
  r=self.ce_weight*c['retain_ce']+(self.alpha*c['retain_repr'] if self.variant!='sim' else 0)
  loss=f+r
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite combined loss')
  self.last_loss_record={'forget_loss':float(f.detach()),'retain_loss':float(r.detach()),'total_loss':float(loss.detach()),'input_tokens':sum(int(clean[k]['attention_mask'].sum()) for k in clean),'loss_tokens':sum(int((clean[k]['labels']!=-100).sum()) for k in clean),'sampled_indices':{k:inputs[k]['index'].detach().cpu().tolist() for k in clean}}
  if model.training:
   self.step_loss_records.append(self.last_loss_record)
   self.component_records.append({'step':self.state.global_step,'components':{k:float(v.detach()) for k,v in c.items()}})
  return (loss,fo) if return_outputs else loss
