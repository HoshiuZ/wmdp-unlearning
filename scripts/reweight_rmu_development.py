"""Reaggregate archived predictions with fixed official subject sizes; no inference."""
import json,csv
from pathlib import Path
from rmu_adaptation_utils import ROOT,OUT,base_samples,dump
BIO_SUBJECTS={'mmlu_anatomy','mmlu_clinical_knowledge','mmlu_college_biology','mmlu_college_medicine','mmlu_high_school_biology','mmlu_human_aging','mmlu_medical_genetics','mmlu_nutrition','mmlu_professional_medicine','mmlu_virology'}
source=base_samples()['mmlu'];weights={s:len(r) for s,r in source.items()};assert sum(weights.values())==14042

def summarize(rows):
 per={}
 for r in rows:per.setdefault(r['task'],[]).append(r)
 assert set(per)==set(weights)
 def group(names):
  names=sorted(names);n=sum(weights[s] for s in names)
  return {'subjects':len(names),'full_test_questions':n,'equal_percent':100*sum(sum(r['acc'] for r in per[s])/len(per[s]) for s in names)/len(names),'weighted_percent':100*sum(weights[s]*sum(r['acc'] for r in per[s])/len(per[s]) for s in names)/n}
 return {'all':group(weights),'biology_related':group(BIO_SUBJECTS),'other':group(set(weights)-BIO_SUBJECTS),'per_subject':{s:{'dev_n':len(rs),'correct':sum(r['acc'] for r in rs),'accuracy_percent':100*sum(r['acc'] for r in rs)/len(rs),'full_weight_n':weights[s]} for s,rs in per.items()}}

def main():
 paths=[OUT/'original_development_protocol_check/samples.json']+sorted(OUT.glob('*screen40/checkpoint-*/development/samples.json'))
 newer=ROOT/'open-unlearning/saves/unlearn/rmu_scale_v2'
 paths+=sorted(newer.glob('*/checkpoint-*/development/samples.json')) if newer.exists() else []
 rows=[];baseline=None
 for p in paths:
  data=json.loads(p.read_text());m=summarize(data['mmlu']);b=100*sum(r['acc'] for r in data['wmdp_bio'])/len(data['wmdp_bio'])
  if 'original_development' in str(p):baseline=m
  rows.append({'path':str(p.relative_to(ROOT)),'bio_percent':b,'mmlu':m})
 assert baseline is not None
 for r in rows:
  r['equal_change_pp']=r['mmlu']['all']['equal_percent']-baseline['all']['equal_percent'];r['weighted_change_pp']=r['mmlu']['all']['weighted_percent']-baseline['all']['weighted_percent']
  for group in ['biology_related','other']:r['mmlu'][group]['weighted_change_pp']=r['mmlu'][group]['weighted_percent']-baseline[group]['weighted_percent']
  r['pareto_weighted']=not any(q['bio_percent']<=r['bio_percent'] and q['mmlu']['all']['weighted_percent']>=r['mmlu']['all']['weighted_percent'] and (q['bio_percent']<r['bio_percent'] or q['mmlu']['all']['weighted_percent']>r['mmlu']['all']['weighted_percent']) for q in rows)
 best={str(t):min([r for r in rows if r['weighted_change_pp']>=-t],key=lambda x:(x['bio_percent'],-x['mmlu']['all']['weighted_percent'])) for t in [2,3]}
 dump(ROOT/'logs/rmu_scale_v2_reweighted_development.json',{'weights':weights,'biology_subject_definition':sorted(BIO_SUBJECTS),'baseline':baseline,'rows':rows,'best_at_2_and_3pp':best,'limitation':'Post-stratification uses full test subject sizes; 30 questions per subject remain noisy. Previously observed official test subsets, not independent validation.'})
 with (ROOT/'logs/rmu_scale_v2_development.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['path','bio_percent','mmlu_equal_percent','mmlu_weighted_percent','equal_change_pp','weighted_change_pp','biology_change_pp','other_change_pp','pareto_weighted'])
  for r in rows:w.writerow([r['path'],r['bio_percent'],r['mmlu']['all']['equal_percent'],r['mmlu']['all']['weighted_percent'],r['equal_change_pp'],r['weighted_change_pp'],r['mmlu']['biology_related']['weighted_change_pp'],r['mmlu']['other']['weighted_change_pp'],r['pareto_weighted']])
 print('Weighted baseline',baseline['all']);print('Best', {t:(r['path'],r['bio_percent'],r['weighted_change_pp']) for t,r in best.items()})
if __name__=='__main__':main()
