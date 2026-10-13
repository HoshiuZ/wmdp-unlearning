"""Lock <=2 full evaluations from weighted dev Pareto candidates, no post-full search."""
import json,subprocess
from pathlib import Path
from rmu_adaptation_utils import ROOT,REPO,BASE,load_delta,score,dump

def main():
 path=ROOT/'logs/rmu_scale_v2_full_selection.json';assert not path.exists(),'Selection already locked'
 data=json.loads((ROOT/'logs/rmu_scale_v2_reweighted_development.json').read_text());rows=data['rows'];new=[]
 for r in rows:
  if '/rmu_scale_v2/' not in r['path']:continue
  checkpoint=(ROOT/r['path']).parent.parent;available=(checkpoint/'checkpoint.json').exists()
  r['weights_available']=available
  if available and r['pareto_weighted']:new.append(r)
 selected=[]
 for threshold in [2,3]:
  eligible=[r for r in new if r['weighted_change_pp']>=-threshold]
  if not eligible:continue
  best=min(eligible,key=lambda r:(r['bio_percent'],-r['mmlu']['all']['weighted_percent']))
  if best['path'] not in [r['path'] for r in selected]:selected.append(best)
 dump(path,{'selected':selected,'all_new_weighted_frontier_with_weights':new,'global_best_at_2_and_3pp':data['best_at_2_and_3pp'],'rule':'At most two new saved non-dominated checkpoints, best under2 and under3 weighted-development loss; deduplicate. No full result used to change this selection.','independent_test':False,'max_full_evaluations':2,'note':'Original official tests and prior full statistics already used for exploration; full results remain exploratory. First-trial10/20 weights absent and not eligible.'});print([(r['path'],r['bio_percent'],r['weighted_change_pp']) for r in selected],flush=True)

if __name__=='__main__':main()
