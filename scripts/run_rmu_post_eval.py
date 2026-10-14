"""Evaluate the fixed final RMU checkpoint against archived baseline tasks."""
import os,sys,json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT/'open-unlearning'
sys.path.insert(0,str(REPO/'src')); os.chdir(REPO)
import torch,yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from omegaconf import OmegaConf
from evals.lm_eval import LMEvalEvaluator

def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def samples(run,task):
 return json.loads((REPO/'saves/eval'/run/'LMEval_EVAL.json').read_text())[task]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base-check',action='store_true');args=parser.parse_args()
 baseline=samples('qwen3_4b_base_bio_full_v1','wmdp_bio')['wmdp_bio']
 assert [s['doc_id'] for s in baseline]==list(range(1273))
 snapshot=ROOT/'data/evaluation/wmdp_bio/test.parquet';snapshot.parent.mkdir(parents=True,exist_ok=True)
 if not snapshot.exists():Dataset.from_list([s['doc'] for s in baseline]).to_parquet(str(snapshot))
 taskdir=REPO/'configs/eval/tasks/wmdp_pinned';taskdir.mkdir(parents=True,exist_ok=True)
 import lm_eval
 official=Path(lm_eval.__file__).parent/'tasks/wmdp'
 template=yaml.safe_load((official/'_default_template_yaml').read_text());template.update(dataset_path='parquet',dataset_name=None,dataset_kwargs={'data_files':{'test':'../data/evaluation/wmdp_bio/test.parquet'}})
 (taskdir/'_default_template_yaml').write_text(yaml.safe_dump(template,sort_keys=False))
 (taskdir/'wmdp_bio.yaml').write_text((official/'wmdp_bio.yaml').read_text().replace('"dataset_name": "wmdp-bio"','"dataset_name": null'))
 modelpath=ROOT/'models/Qwen3-4B-Instruct-2507' if args.base_check else REPO/'saves/unlearn/qwen3_4b_rmu_bio_formal80_v1/model'
 tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda')
 protocol=dict(batch_size=1,num_fewshot=0,apply_chat_template=False,system_instruction=None,log_samples=True,random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
 for task in (['wmdp_bio'] if args.base_check else ['wmdp_bio','mmlu']):
  run='qwen3_4b_base_bio_protocol_check_v1' if args.base_check else f'qwen3_4b_rmu_formal80_{task}_0shot_full_v1'
  out=REPO/'saves/eval'/run
  cfg=OmegaConf.create(dict(tasks=[task],task_include_path=str(taskdir if task=='wmdp_bio' else REPO/'configs/eval/tasks/mmlu_pinned'),simple_evaluate_args={**protocol,**({'limit':16} if args.base_check else {})},overwrite=False,output_dir=str(out)))
  dump(out/'evaluation_config.json',dict(model_path=str(modelpath),task=task,protocol=OmegaConf.to_container(cfg),snapshot_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest() if task=='wmdp_bio' else None))
  LMEvalEvaluator(cfg).evaluate(model,tokenizer=tokenizer)
  actual=samples(run,task);old=samples('qwen3_4b_base_bio_full_v1' if task=='wmdp_bio' else 'qwen3_4b_base_mmlu_0shot_full_v1',task)
  checks=[]
  for name,rows in actual.items():
   prior={s['doc_id']:s for s in old[name]}
   for row in rows:
    ref=prior[row['doc_id']]
    assert row['doc']==ref['doc'] and row['arguments']==ref['arguments'],(name,row['doc_id'],'protocol mismatch')
    if args.base_check:assert row['acc']==ref['acc'],(name,row['doc_id'],'baseline prediction mismatch')
   checks.append(dict(task=name,n=len(rows),correct=sum(s['acc'] for s in rows),baseline_correct=sum(prior[s['doc_id']]['acc'] for s in rows)))
  dump(out/'comparability_check.json',dict(exact_documents_and_scoring_arguments=True,baseline_prediction_check=args.base_check,subjects=checks,total=sum(x['n'] for x in checks)))
  print('COMPLETED',run,checks if task=='wmdp_bio' else {'total':sum(x['n'] for x in checks),'correct':sum(x['correct'] for x in checks)},flush=True)
if __name__=='__main__':main()
