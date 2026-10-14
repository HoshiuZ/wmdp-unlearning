"""RMU activation forgetting plus representation and CE retention."""
import torch
import torch.nn.functional as F
from trainer.unlearn.rmu import RMU

def token_nll(logits,labels):
 target=labels[:,1:];valid=target!=-100;counts=valid.sum(-1)
 if (counts==0).any():raise ValueError('No effective prediction tokens')
 values=F.cross_entropy(logits[:,:-1].float().transpose(1,2),target,reduction='none',ignore_index=-100)
 return values.sum(-1)/counts

class RMUCE(RMU):
 def __init__(self,ce_weight=0.09004336292634454,*args,**kwargs):
  super().__init__(*args,**kwargs);self.ce_weight=ce_weight;self.component_records=[]
 def components(self,model,inputs):
  clean={role:{key:inputs[role][key] for key in ['input_ids','attention_mask','labels']} for role in ['forget','retain']}
  f=clean['forget'];r=clean['retain'];fact,fo=self.forward_with_cache(model,f,self.model_module,no_grad=False)
  target=self.get_control_vector(fact.shape[-1]).to(fact).expand_as(fact)
  result={'rmu_forget':self.compute_activation_loss(fact,target,(f['labels']!=-100)&f['attention_mask'].bool())}
  ract,ro=self.forward_with_cache(model,r,self.model_module,no_grad=False)
  result['retain_ce']=token_nll(ro.logits,r['labels']).mean()
  ref,_=self.forward_with_cache(self.ref_model,r,self.ref_module,no_grad=True)
  result['retain_repr']=self.compute_activation_loss(ract,ref.to(ract),(r['labels']!=-100)&r['attention_mask'].bool())
  return result,fo,clean
 def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
  c,fo,clean=self.components(model,inputs);f=c['rmu_forget'];r=self.ce_weight*c['retain_ce']+self.alpha*c['retain_repr'];loss=f+r
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RMU+CE loss')
  self.last_loss_record={'forget_loss':float(f.detach()),'retain_loss':float(r.detach()),'total_loss':float(loss.detach()),'input_tokens':sum(int(clean[k]['attention_mask'].sum()) for k in clean),'loss_tokens':sum(int((clean[k]['labels']!=-100).sum()) for k in clean),'sampled_indices':{k:inputs[k]['index'].detach().cpu().tolist() for k in clean}}
  if model.training:
   self.step_loss_records.append(self.last_loss_record);self.component_records.append({'step':self.state.global_step,'effective_prediction_tokens':sum(int((clean[k]['labels'][:,1:]!=-100).sum()) for k in clean),'components':{k:float(v.detach()) for k,v in c.items()}})
  return (loss,fo) if return_outputs else loss
