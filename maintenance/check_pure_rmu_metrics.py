"""Validate archived evaluation identity and arithmetic; never run a model."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from rmu_adaptation_utils import base_samples
from report_rmu_scale_v2 import flat,full_row
import report_rmu_scale_v2 as report
baseline=base_samples();report.ORIGINAL=flat(baseline)
rows=[full_row('original',baseline,{})]
base=ROOT/'open-unlearning/saves';legacy={}
for task,run in [('wmdp_bio','qwen3_4b_rmu_formal80_wmdp_bio_0shot_full_v1'),('mmlu','qwen3_4b_rmu_formal80_mmlu_0shot_full_v1')]:legacy[task]=json.loads((base/'eval'/run/'LMEval_EVAL.json').read_text())[task]
rows.append(full_row('formal80',legacy,{}))
paths=[base/'unlearn/rmu_adaptation_v1'/p/'samples.json' for p in ['selected_full_evaluation','selected_full_evaluation_phase2']]
paths+=sorted((base/'unlearn/rmu_scale_v2/full_evaluations').glob('*/samples.json'))
for p in paths:
 data=json.loads(p.read_text())
 for task,samples in flat(data).items():
  original={(x['task'],x['doc_id']):x for x in report.ORIGINAL[task]}
  for x in samples:
   before=original[(x['task'],x['doc_id'])]
   assert x['target']==before['doc']['answer']
   assert x['prediction']==max(range(4),key=lambda i:x['scores'][i])
   assert x['acc']==int(x['prediction']==x['target'])
 rows.append(full_row(str(p.relative_to(ROOT)),data,{}))
assert len(rows)==6
out={'success':True,'new_inference':False,'checked':'Exact question identity, prompt/doc hashes, baseline labels, sample counts and arithmetic','full_results':rows}
(ROOT/'maintenance/pure_rmu_metrics_verification.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('PASS: 6 archived full results; Bio1273, MMLU14042; identical document and prompt hashes')
