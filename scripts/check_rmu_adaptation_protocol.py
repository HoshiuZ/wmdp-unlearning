"""Verify archived-request evaluator on original model before selecting candidates."""
import os,json
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import BASE,ROOT,REPO,OUT,score,dump
os.chdir(REPO)
model=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
out=OUT/'original_development_protocol_check';assert not out.exists()
metrics=score(model,tok,out)
rows=json.loads((out/'samples.json').read_text());assert all(s['acc']==s['baseline_acc'] for subjects in rows.values() for s in subjects)
dump(out/'check.json',{'all_1966_predictions_match_original_baseline':True,'metrics':metrics})
print('PROTOCOL_PASS',metrics,flush=True)
