"""Sequential evaluation of locked saved candidates; no training or further selection."""
import os,json,time,subprocess
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,load_delta,score

def main():
 selection=json.loads((ROOT/'logs/simnpo_rmu_full_selection.json').read_text());assert len(selection['selected'])<=2
 out=REPO/'saves/unlearn/simnpo_rmu_v1/full_evaluations';out.mkdir(exist_ok=True)
 os.chdir(REPO)
 for row in selection['selected']:
  checkpoint=(ROOT/row['path']).parent.parent;key=checkpoint.parent.name+'_'+checkpoint.name;dest=out/key
  assert not dest.exists(),'Do not overwrite/repeat completed or partial full evaluation'
  model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);load_delta(model,checkpoint)
  dump(dest/'evaluation_config.json',{'checkpoint':str(checkpoint.relative_to(ROOT)),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'task_samples':{'wmdp_bio':1273,'mmlu':14042},'num_fewshot':0,'batch_size':1,'dtype':'bfloat16','attn':'sdpa','apply_chat_template':False,'system_instruction':None,'scoring':'Exact archived original requests, argmax continuation loglikelihood; no stochastic generation','exploratory':True})
  metrics=score(model,tok,dest,full=True);dump(dest/'result.json',{'metrics':metrics,'checkpoint':str(checkpoint.relative_to(ROOT)),'loss_le_2pp':metrics['mmlu']['change_percentage_points']>=-2,'loss_le_3pp':metrics['mmlu']['change_percentage_points']>=-3,'no_general_success_claim':True})
  print('FULL_COMPLETE',key,metrics,flush=True);del model,tok
  import gc;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
