"""Reuse exact input audits; audit cached original/RMU replies with official output template."""
import argparse,time
from .common import *

def main():
 import torch
 from safety.qwen3guard.moderator import Qwen3Guard
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args()
 p=protocol();assert sha((ROOT/CFG['guard_config']).read_bytes())==p['guard_config_sha256'];assert sha((ROOT/'safety/qwen3guard/moderator.py').read_bytes())==p['guard_parser_sha256']
 guard=Qwen3Guard();assert guard.runtime==p['guard_runtime'];torch.cuda.reset_peak_memory_stats();c=db()
 old=sqlite3.connect(ROOT/'cache/wildguardtest'/p['dataset_revision']/p['prior_cache_fingerprint']/'audits.sqlite')
 start=time.perf_counter();session=dict(phase=args.phase,input_reused=0,output_new=0,output_reused=0,output_not_needed=0,started_unix=time.time(),sources={f:sha((HERE/f).read_bytes()) for f in ['audit.py','common.py']})
 dest=LOCAL/'sessions'/f'audit_{args.phase}_{time.time_ns()}.json';streak=0
 for index,row in enumerate(select(records(),args.phase)):
  ik=key('input_audit',row,p['guard_revision']);a=get(c,ik)
  if a is None:
   assert row['content_sha256']==digest([row['request']])
   oldkey=digest(dict(task='input',content_sha256=row['content_sha256'],fingerprint=p['prior_cache_fingerprint'],revision=p['guard_revision']))
   cached=old.execute('SELECT result FROM audits WHERE key=?',(oldkey,)).fetchone();assert cached is not None
   a=json.loads(cached[0]);assert a['task']=='input'
   text=guard.tokenizer.apply_chat_template([{'role':'user','content':row['request']}],tokenize=False)
   assert sha(text.encode())==a['rendered_input_sha256'];put(c,ik,'input_audit',row['sample_id'],a)
  session['input_reused']+=1
  needed=any(not a['policies'][policy]['blocked'] for policy in CFG['policies'])
  for model in ('original','rmu80'):
   fp=p['base_fingerprint' if model=='original' else 'rmu80_fingerprint'];g=get(c,key('generation',row,fp));assert g is not None
   if not needed or g['status']!='ok':session['output_not_needed']+=1;continue
   content=digest([row['request'],g['response']]);ok=key('output_audit',row,fp,content);audit=get(c,ok)
   if audit is None:
    audit=guard.moderate_output(row['request'],g['response']);put(c,ok,'output_audit',row['sample_id'],audit);session['output_new']+=1
    streak=streak+1 if audit['status']!='ok' else 0
   else:session['output_reused']+=1
   if streak>=CFG['max_consecutive_failures']:dump(dest,session);raise RuntimeError('Sustained output audit errors; stop')
  session.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),after_resources=resource())
  dump(dest,session)
  if (index+1)%50==0:print(json.dumps(dict(phase=args.phase,completed=index+1,output_new=session['output_new'])),flush=True)
 session['complete']=True;dump(dest,session);print('AUDIT_COMPLETE',args.phase,session['output_new'],flush=True)
if __name__=='__main__':main()
