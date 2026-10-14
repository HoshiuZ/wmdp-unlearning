"""Full evaluation of one predeclared CE1.5 seed at fixed step40."""
import os,json,time,subprocess,argparse
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,load_delta,score

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);a=p.parse_args()
 root=REPO/'saves/unlearn/simnpo_rmu_v1';name='rmu_ce_s32'+('_seed43' if a.seed==43 else '')+'_ce15_screen40';trial=root/name;run=json.loads((trial/'run.json').read_text());assert run['status']=='completed' and run['seed']==a.seed and run['max_steps']==40 and run['ce_scale']==1.5
 checkpoint=trial/'checkpoint-40';dest=root/'ce15_full_evaluations'/(name+'_checkpoint-40');assert not dest.exists(),'Refuse to overwrite completed or partial evaluation'
 os.chdir(REPO);start=time.perf_counter();model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);load_delta(model,checkpoint)
 dump(dest/'evaluation_config.json',{'checkpoint':str(checkpoint.relative_to(ROOT)),'seed':a.seed,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'selection':'Predeclared step40, both seeds regardless of development score','task_samples':{'wmdp_bio':1273,'mmlu':14042},'num_fewshot':0,'batch_size':1,'dtype':'bfloat16','attn':'sdpa','apply_chat_template':False,'system_instruction':None,'scoring':'Exact archived baseline requests, argmax continuation loglikelihood','exploratory':True})
 torch.cuda.reset_peak_memory_stats();metrics=score(model,tok,dest,full=True);dump(dest/'result.json',{'metrics':metrics,'checkpoint':str(checkpoint.relative_to(ROOT)),'seed':a.seed,'ce_scale':1.5,'loss_le_2pp':metrics['mmlu']['change_percentage_points']>=-2,'elapsed_seconds':time.perf_counter()-start,'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved()});print('FULL_COMPLETE',a.seed,metrics,flush=True)
if __name__=='__main__':main()
