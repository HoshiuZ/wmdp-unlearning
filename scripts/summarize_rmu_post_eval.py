"""Validate complete paired evaluations and archive percentage-point changes."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'open-unlearning/saves/eval'
def load(p):return json.loads(p.read_text())
def main():
 rows=[]
 for task,base,n in [('wmdp_bio','qwen3_4b_base_bio_full_v1',1273),('mmlu','qwen3_4b_base_mmlu_0shot_full_v1',14042)]:
  post=f'qwen3_4b_rmu_formal80_{task}_0shot_full_v1'
  a=load(E/base/'LMEval_EVAL.json')[task];b=load(E/post/'LMEval_EVAL.json')[task]
  assert a.keys()==b.keys()
  before=after=total=lost=gained=0
  for subject,records in b.items():
   refs={x['doc_id']:x for x in a[subject]};assert len(refs)==len(records)
   for x in records:
    y=refs[x['doc_id']];assert x['doc']==y['doc'] and x['arguments']==y['arguments']
    before+=int(y['acc']);after+=int(x['acc']);total+=1
    lost+=int(y['acc']==1 and x['acc']==0);gained+=int(y['acc']==0 and x['acc']==1)
  assert total==n
  full=load(E/post/f'LMEval_FULL_{task}.json');assert all(v==0 for v in full['n-shot'].values())
  rows.append(dict(task=task,samples=n,subjects=len(b),baseline_correct=before,post_correct=after,baseline_accuracy_percent=100*before/n,post_accuracy_percent=100*after/n,baseline_standard_error_percentage_points=100*load(E/base/'LMEval_SUMMARY.json')[task+'/acc_stderr'],post_standard_error_percentage_points=100*load(E/post/'LMEval_SUMMARY.json')[task+'/acc_stderr'],change_percentage_points=100*(after-before)/n,correct_to_wrong=lost,wrong_to_correct=gained,baseline_directory=str((E/base).relative_to(ROOT)),post_directory=str((E/post).relative_to(ROOT)),exact_documents_and_scoring_arguments=True))
 check=load(E/'qwen3_4b_base_bio_protocol_check_v1/LMEval_EVAL.json')['wmdp_bio']['wmdp_bio']
 baseline=load(E/'qwen3_4b_base_bio_full_v1/LMEval_EVAL.json')['wmdp_bio']['wmdp_bio']
 score_delta=max(abs(float(x['resps'][k][0][0])-float(baseline[x['doc_id']]['resps'][k][0][0])) for x in check for k in range(4))
 report=dict(script_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/run_rmu_post_eval.py',ROOT/'scripts/summarize_rmu_post_eval.py',ROOT/'open-unlearning/src/evals/lm_eval.py']},original_model_protocol_check_max_loglikelihood_difference=score_delta,commands=['HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 envs/unlearning/bin/python scripts/run_rmu_post_eval.py --base-check','HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 envs/unlearning/bin/python scripts/run_rmu_post_eval.py'],model_id='qwen3_4b_rmu_bio_formal80_v1',recorded_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),checkpoint_step=80,checkpoint_selection='fixed in advance; no test-set selection',evaluations=rows,interpretation='WMDP accuracy is a proxy for targeted capability; decrease does not establish complete knowledge removal. MMLU measures utility on this benchmark only.',protocol_check='Original-model first 16 Bio predictions agree with archived baseline; all post-evaluation documents and scoring arguments match baseline.',artifacts={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for r in rows for p in (ROOT/r['post_directory']).glob('*.json')})
 (ROOT/'logs/rmu_post_eval_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
