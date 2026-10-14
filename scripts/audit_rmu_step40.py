"""Read-only comparison of step40 tensors and historical runtime evidence."""
import gzip,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from rmu_adaptation_utils import ROOT,BASE,dump,materialized,tensor_hash
from rmu_step_common import RUN,HISTORY,training_rows
index=json.loads((BASE/'model.safetensors.index.json').read_text())['weight_map'];old=json.loads((HISTORY/'checkpoint-40/checkpoint.json').read_text());new=json.loads((RUN/'checkpoint-40/checkpoint.json').read_text());comparisons=[]
for a,b in zip(old['parameters'],new['parameters']):
 assert a['name']==b['name'] and a['base_tensor_sha256']==b['base_tensor_sha256']
 with safe_open(BASE/index[a['name']],framework='pt',device='cpu') as f:original=f.get_tensor(a['name'])
 assert tensor_hash(original)==a['base_tensor_sha256'];restored=[]
 for parent,row in [(HISTORY/'checkpoint-40',a),(RUN/'checkpoint-40',b)]:
  file=materialized(parent/row['file']);assert hashlib.sha256(file.read_bytes()).hexdigest()==row['compressed_sha256']
  delta=np.frombuffer(gzip.decompress(file.read_bytes()),dtype=np.uint16).reshape(row['shape']);t=torch.from_numpy(np.bitwise_xor(original.view(torch.uint16).numpy(),delta).copy()).view(torch.bfloat16);assert tensor_hash(t)==row['trained_tensor_sha256'];restored.append(t)
 diff=(restored[0].float()-restored[1].float()).abs();comparisons.append({'parameter':a['name'],'unequal_elements':int((diff!=0).sum()),'total_elements':diff.numel(),'max_abs_difference':float(diff.max()),'mean_abs_difference':float(diff.mean())})
rows=training_rows(RUN/'training_metrics.jsonl');oldrows=training_rows(HISTORY/'training_metrics.jsonl');keys=['forget_loss','retain_loss','total_loss','gradient_norm'];errors={k:max(abs(a[k]-b[k]) for a,b in zip(rows,oldrows)) for k in keys}
sourcefiles=['open-unlearning/src/trainer/unlearn/rmu.py','open-unlearning/src/trainer/unlearn/grad_diff.py','open-unlearning/src/data/unlearn.py','open-unlearning/src/data/pretraining.py','scripts/rmu_adaptation_utils.py','scripts/rmu_validation.py','scripts/run_rmu_formal.py']
source_changes=subprocess.check_output(['git','diff','147a315','ff618f4','--',*sourcefiles],cwd=ROOT,text=True);assert not source_changes
result={'same_training_core_sources_as_historical_commit':True,'source_paths_compared':sourcefiles,'same_pinned_base_tokenizer_and_data':True,'configuration_comparison':'preflight normalized equality; only length and output/description metadata differ','same_sampling_and_input_loss_token_budgets':json.loads((RUN/'reproduction40.json').read_text())['sampling_token_budget_and_direction_match'],'maximum_step_loss_or_gradient_norm_difference':errors,'parameter_differences':comparisons,'numerical_probe':'logs/rmu_step_sdpa_numerics_probe.json','historical_environment_limit':'Historical config/library versions available; exact low-level kernel scheduling not archived. No claim to reconstruct a particular historical CUDA launch.'}
dump(RUN/'reproduction40_numerical_audit.json',result);print(json.dumps(result,indent=2))
