"""Evaluate a retained checkpoint using exact archived original scoring requests."""
import argparse,os,json,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_ce_utils import ROOT,REPO,BASE,OUT,dump,load_delta,score

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[42,43],required=True);p.add_argument('--step',type=int,choices=[10,20,40],default=40);p.add_argument('--output',required=True);p.add_argument('--development',action='store_true');a=p.parse_args();dest=(ROOT/a.output).resolve();assert dest.is_relative_to(ROOT) and not dest.exists(),'Use a new output directory within the project';checkpoint=OUT/f'seed{a.seed}/checkpoint-{a.step}';os.chdir(REPO)
 model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);load_delta(model,checkpoint);metrics=score(model,tok,dest,full=not a.development);dump(dest/'result.json',{'seed':a.seed,'checkpoint':str(checkpoint.relative_to(ROOT)),'metrics':metrics});print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
