"""Select by frozen development scores; evaluate only one locked candidate in full."""
import os,json,argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,OUT,dump,score,load_delta

def candidates():
 rows=[]
 for p in sorted(OUT.glob('*_screen40/run.json')):
  run=json.loads(p.read_text())
  if run['status']!='completed':continue
  for d in run['development']:
   m=d['metrics'];rows.append({'experiment':run['name'],'hook':run['hook'],'update_parameters':run['selected_parameters'],'learning_rate':run['learning_rate'],'alpha':run['alpha'],'steering_coefficient':run['steering_coefficient'],'step':d['step'],'checkpoint':d['checkpoint'],'wmdp_bio_development_percent':m['wmdp_bio']['accuracy_percent'],'mmlu_development_percent':m['mmlu']['accuracy_percent'],'wmdp_bio_change_pp':m['wmdp_bio']['change_percentage_points'],'mmlu_change_pp':m['mmlu']['change_percentage_points'],'development_constraint_pass':m['mmlu']['change_percentage_points']>=-2,'peak_allocated_bytes':run['cuda_peak_allocated_bytes'],'training_wall_seconds':run['elapsed_seconds']})
 return rows

def choose(rows):
 passing=[r for r in rows if r['development_constraint_pass']]
 if passing:return min(passing,key=lambda r:(r['wmdp_bio_development_percent'],-r['mmlu_development_percent'],r['step'],r['experiment']))
 return max(rows,key=lambda r:(r['mmlu_development_percent'],-r['wmdp_bio_development_percent'],-r['step']))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--select-layer',action='store_true');parser.add_argument('--need-fallback',action='store_true');parser.add_argument('--full',action='store_true');a=parser.parse_args()
 rows=candidates();assert rows
 if a.select_layer:
  pick=choose([r for r in rows if r['alpha']==1 and r['learning_rate']==5e-5 and r['steering_coefficient']==2]);dump(ROOT/'logs/rmu_adaptation_layer_selection.json',{'selected':pick,'basis':'frozen development subset constraint, then lowest Bio; if no passing candidate least MMLU loss','all_layer_candidates':rows});print(pick['hook']);return
 if a.need_fallback:print(int(not any(r['development_constraint_pass'] for r in rows)));return
 pick=choose(rows);selection={'selected':pick,'development_constraint_pass':pick['development_constraint_pass'],'all_candidates':rows,'independent_final_test':False,'full_evaluation_label':'exploratory; official test statistics were previously observed, development subset used in tuning'}
 dump(ROOT/'logs/rmu_adaptation_selection.json',selection)
 if a.full and not pick['development_constraint_pass']:
  dump(ROOT/'logs/rmu_adaptation_result.json',{'selection':selection,'full_metrics':None,'full_mmlu_constraint_pass':False,'note':'No development candidate met the 2pp constraint; no full candidate evaluation performed.'});print('NO_QUALIFYING_CANDIDATE',flush=True);return
 if a.full:
  out=OUT/'selected_full_evaluation'
  if out.exists():raise RuntimeError('Refuse to rerun/overwrite selected full evaluation')
  os.chdir(REPO);model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',local_files_only=True);tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
  load_delta(model,ROOT/pick['checkpoint']);metrics=score(model,tok,out,full=True)
  result={'selection':selection,'full_metrics':metrics,'full_mmlu_constraint_pass':metrics['mmlu']['change_percentage_points']>=-2,'full_evaluation_label':'exploratory','success_standard_note':'2pp is this experiment goal, not universal proof of unlearning'}
  dump(ROOT/'logs/rmu_adaptation_result.json',result);print(json.dumps(result['full_metrics']),flush=True)

if __name__=='__main__':main()
