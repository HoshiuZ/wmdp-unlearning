"""Reaggregate fixed development predictions and lock bounded full selection."""
import json,csv,argparse
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,dump
from reweight_rmu_development import summarize
OUT=REPO/'saves/unlearn/simnpo_rmu_v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--lock',action='store_true');a=p.parse_args();old=json.loads((ROOT/'logs/rmu_scale_v2_reweighted_development.json').read_text());rows=list(old['rows']);base=old['baseline']
 for path in sorted(OUT.glob('*screen40/checkpoint-*/development/samples.json')):
  d=json.loads(path.read_text());m=summarize(d['mmlu']);b=100*sum(r['acc'] for r in d['wmdp_bio'])/len(d['wmdp_bio']);r={'path':str(path.relative_to(ROOT)),'bio_percent':b,'mmlu':m,'equal_change_pp':m['all']['equal_percent']-base['all']['equal_percent'],'weighted_change_pp':m['all']['weighted_percent']-base['all']['weighted_percent']}
  for group in ['biology_related','other']:m[group]['weighted_change_pp']=m[group]['weighted_percent']-base[group]['weighted_percent']
  rows.append(r)
 for r in rows:r['pareto_weighted']=not any(q['bio_percent']<=r['bio_percent'] and q['mmlu']['all']['weighted_percent']>=r['mmlu']['all']['weighted_percent'] and (q['bio_percent']<r['bio_percent'] or q['mmlu']['all']['weighted_percent']>r['mmlu']['all']['weighted_percent']) for q in rows)
 best={str(t):min([r for r in rows if r['weighted_change_pp']>=-t],key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent'])) for t in [2,3]};new=[r for r in rows if 'simnpo_rmu_v1' in r['path']]
 result={'rows':rows,'best_all_at_2_and_3pp':best,'biology_subject_definition':old['biology_subject_definition'],'weights':old['weights'],'exploratory':True,'independent_test':False};dump(ROOT/'logs/simnpo_rmu_development.json',result)
 with (ROOT/'logs/simnpo_rmu_development.csv').open('w') as f:
  w=csv.writer(f,lineterminator='\n');w.writerow(['path','bio_percent','mmlu_equal','mmlu_weighted','equal_change_pp','weighted_change_pp','bio_subject_change_pp','other_subject_change_pp','pareto'])
  for r in rows:w.writerow([r['path'],r['bio_percent'],r['mmlu']['all']['equal_percent'],r['mmlu']['all']['weighted_percent'],r['equal_change_pp'],r['weighted_change_pp'],r['mmlu']['biology_related']['weighted_change_pp'],r['mmlu']['other']['weighted_change_pp'],r['pareto_weighted']])
 print('NEW',[(r['path'].split('/')[-4:-2],r['bio_percent'],r['weighted_change_pp'],r['pareto_weighted']) for r in new]);print('BEST_ALL',[(t,r['path'],r['bio_percent'],r['weighted_change_pp']) for t,r in best.items()])
 if a.lock:
  assert len(new)==9
  for mode in ['sim','rmu_ce','joint']:assert json.loads((OUT/(mode+'_screen40')/'run.json').read_text())['status']=='completed'
  selected=[];bestnew={}
  for t in [2,3]:
   eligible=[r for r in new if r['weighted_change_pp']>=-t and r['pareto_weighted']];r=min(eligible,key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent'])) if eligible else None;bestnew[str(t)]=r
   if r and r['path'] not in [x['path'] for x in selected]:selected.append(r)
  assert len(selected)<=2;dest=ROOT/'logs/simnpo_rmu_full_selection.json';assert not dest.exists();dump(dest,{'selected':selected,'best_new_frontier_at_2_and_3pp':bestnew,'rule':'Saved new checkpoints, nondominated by any archived development point, best Bio under2/3 weighted MMLU loss; deduplicate; max2; no full feedback','exploratory':True});print('LOCKED',len(selected))
if __name__=='__main__':main()
