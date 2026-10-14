"""Independent-process repeated BF16 SDPA backward; no model or optimizer update."""
import hashlib,json
from pathlib import Path
import torch
from torch.nn.functional import scaled_dot_product_attention
ROOT=Path(__file__).resolve().parents[1]
def digest(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
torch.manual_seed(123)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
q=torch.randn(1,32,256,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
k=torch.randn(1,8,256,128,device='cuda',dtype=torch.bfloat16,requires_grad=True);v=torch.randn_like(k,requires_grad=True);upstream=torch.randn_like(q)
rows=[];first=None;maxdiff=[0.,0.,0.]
for i in range(20):
 q.grad=k.grad=v.grad=None
 y=scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True,enable_gqa=True)
 y.backward(upstream)
 grads=[x.grad.detach().clone() for x in [q,k,v]]
 if first is None:first=grads
 for j in range(3):maxdiff[j]=max(maxdiff[j],float((grads[j].float()-first[j].float()).abs().max()))
 rows.append({'output_sha256':digest(y),'gradient_sha256':[digest(g) for g in grads]})
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
 q.grad=k.grad=v.grad=None;scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True,enable_gqa=True).backward(upstream)
result={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'dtype':'bfloat16','query_shape':list(q.shape),'key_value_shape':list(k.shape),'enable_gqa':True,'dropout':0,'causal':True,'identical_inputs_and_upstream_gradient':True,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'backend_events':[e.key for e in prof.key_averages() if 'scaled_dot' in e.key],'repeats':rows,'unique_forward_hashes':len({r['output_sha256'] for r in rows}),'unique_gradient_triplets':len({tuple(r['gradient_sha256']) for r in rows}),'max_abs_gradient_difference_from_first':maxdiff,'no_model_or_optimizer_update':True}
(ROOT/'logs/rmu_step_sdpa_numerics_probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='repeats'},indent=2))
