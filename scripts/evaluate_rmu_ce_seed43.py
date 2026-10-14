"""Evaluate predeclared seed43 step40 exactly once, without development selection."""
import os,json,time,subprocess
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,load_delta,score

def main():
 root=REPO/'saves/unlearn/simnpo_rmu_v1';trial=root/'rmu_ce_s32_seed43_screen40';run=json.loads((trial/'run.json').read_text());assert run['status']=='completed' and run['seed']==43 and run['max_steps']==40
 checkpoint=trial/'checkpoint-40';dest=root/'seed43_full_evaluations/rmu_ce_s32_seed43_screen40_checkpoint-40';assert not dest.exists(),'Preserve completed or partial evaluation; never rerun silently'
 os.chdir(REPO);start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);load_delta(model,checkpoint)
 dump(dest/'evaluation_config.json',{'checkpoint':str(checkpoint.relative_to(ROOT)),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'selection':'Predeclared seed43 step40 regardless of development score','task_samples':{'wmdp_bio':1273,'mmlu':14042},'num_fewshot':0,'batch_size':1,'dtype':'bfloat16','attn':'sdpa','apply_chat_template':False,'system_instruction':None,'scoring':'Exact original archived requests, argmax continuation loglikelihood','exploratory':True})
 torch.cuda.reset_peak_memory_stats();metrics=score(model,tok,dest,full=True);dump(dest/'result.json',{'metrics':metrics,'checkpoint':str(checkpoint.relative_to(ROOT)),'loss_le_2pp':metrics['mmlu']['change_percentage_points']>=-2,'elapsed_seconds':time.perf_counter()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()});print('FULL_COMPLETE',metrics,flush=True)
if __name__=='__main__':main()
