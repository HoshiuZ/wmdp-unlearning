import torch
from .common import *
from safety.qwen3guard.moderator import Qwen3Guard

def main():
 c=connection();p=old.protocol();assert sha((ROOT/CFG['guard_config']).read_bytes())==manifest()['guard_config_sha256'];assert sha((ROOT/'safety/qwen3guard/moderator.py').read_bytes())==manifest()['guard_parser_sha256']
 guard=Qwen3Guard();assert guard.runtime==p['guard_runtime'];torch.cuda.reset_peak_memory_stats();start=time.perf_counter();session={'new':0,'reused_old':0,'reused_diagnostic':0}
 previous=sqlite3.connect('file:'+str(old.LOCAL/'records.sqlite')+'?mode=ro',uri=True);streak=0
 for model in ('original','rmu80'):
  fp=manifest()['generator_fingerprints'][model]
  for r in rows():
   for length in (512,1024):
    k=key('audit'+str(length),model,r);a=get(c,k)
    if a is not None:session['reused_diagnostic']+=1;continue
    g=oldgen(r,model) if length==512 else get(c,key('generate1024',model,r));assert g and g['status']=='ok'
    a=None
    if length==512:
     a=old.get(previous,old.key('output_audit',r,fp,digest([r['request'],g['response']])))
    if a is None:
     a=guard.moderate_output(r['request'],g['response']);session['new']+=1
     streak=streak+1 if a['status']!='ok' else 0
    else:session['reused_old']+=1
    rendered=guard.tokenizer.apply_chat_template([{'role':'user','content':r['request']},{'role':'assistant','content':g['response']}],tokenize=False)
    assert a['rendered_input_sha256']==sha(rendered.encode());put(c,k,'audit'+str(length),a)
    session.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved());dump(LOCAL/'audit_session.json',session)
    if streak>=5:raise RuntimeError('Sustained audit errors; stop with cached results')
  print('AUDITED',model,flush=True)
 session['complete']=True;dump(LOCAL/'audit_session.json',session)
if __name__=='__main__':main()
