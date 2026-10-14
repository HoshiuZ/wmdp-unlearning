"""Run only the authorized fixed four-group record stage, sequentially; stop on any worker error."""
import os,subprocess,sys,time
from .common import *

def main():
 pilot=json.loads((HERE/'pilot_summary.json').read_text());assert pilot['phase']=='pilot' and pilot['n']==20
 for model in pilot['models'].values():assert model['completed']==20 and model['failed']==0
 stages=[['generate','--model','original','--phase','full'],['generate','--model','rmu80','--phase','full'],['audit','--phase','full'],['finish','--phase','full']]
 state={'started_unix':time.time(),'stages':[]};path=LOCAL/'orchestration.json';dump(path,state)
 for stage in stages:
  name='_'.join(stage).replace('--','');start=time.perf_counter();log=LOCAL/(name+'.log')
  print('START',name,flush=True)
  with log.open('a') as out:r=subprocess.run([sys.executable,'-m','safety.four_groups.'+stage[0],*stage[1:]],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
  state['stages'].append(dict(stage=stage,returncode=r.returncode,wall_seconds=time.perf_counter()-start));dump(path,state)
  if r.returncode:raise RuntimeError('Worker failed: '+name+'; local log retained; no next stage started')
  print('DONE',name,flush=True)
 state.update(complete=True,ended_unix=time.time());dump(path,state)
 print('AUTHORIZED_STAGE_COMPLETE_STOP',flush=True)
if __name__=='__main__':main()
