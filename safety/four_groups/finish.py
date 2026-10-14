"""Exclusive four-group records, blind text list/local mapping, public aggregates only."""
import argparse,random,shutil,time
from collections import Counter
from .common import *
from .policy import outcome

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args()
 rows=select(records(),args.phase);p=protocol();c=db();summary={'phase':args.phase,'n':len(rows),'models':{},'systems':{}}
 blind={};mapping=[];systems=[]
 def blind_entry(row,answer,where):
  pair=digest([row['request'],answer]);blind.setdefault(pair,{'request':row['request'],'answer':answer});mapping.append(dict(content_key=pair,sample_id=row['sample_id'],**where))
 for model,rawgroup,protectedgroup in [('original','A','B'),('rmu80','C','D')]:
  fp=p['base_fingerprint' if model=='original' else 'rmu80_fingerprint'];gs=[]
  for row in rows:
   g=get(c,key('generation',row,fp));assert g is not None;gs.append(g)
   if g['status']=='ok':blind_entry(row,g['response'],dict(group=rawgroup,policy=None,state='raw_answer',generation_ref=key('generation',row,fp)))
  good=[g for g in gs if g['status']=='ok'];summary['models'][model]=dict(completed=len(good),failed=len(gs)-len(good),
    hit_generation_limit=sum(g.get('hit_generation_limit',False) for g in good),limit_fraction_of_completed=sum(g.get('hit_generation_limit',False) for g in good)/len(good) if good else None,
    finish_reasons=dict(Counter(g.get('finish_reason',g.get('error','unknown')) for g in gs)),input_tokens=sum(g.get('input_tokens',0) for g in gs),output_tokens=sum(g.get('output_tokens',0) for g in gs),unexpected_thinking=sum(g.get('contains_think_tags',False) for g in gs))
  for policy in CFG['policies']:
   counts=Counter();bylabel={l:Counter() for l in ('harmful','unharmful')}
   for row,g in zip(rows,gs):
    ik=key('input_audit',row,p['guard_revision']);a=get(c,ik);assert a is not None
    ok=key('output_audit',row,fp,digest([row['request'],g['response']])) if g['status']=='ok' else None
    oa=get(c,ok) if ok else None;state=outcome(a,g,oa,policy);counts[state]+=1;bylabel[row['gold']][state]+=1
    final=g['response'] if state=='allow' else CFG['fixed_refusal']
    item=dict(sample_id=row['sample_id'],group=protectedgroup,policy=policy,state=state,input_label=row['gold'],
              actual_simulated_generator_invocation=state not in ('input_block','exception_input'),
              final_response_ref=key('generation',row,fp) if state=='allow' else 'config.fixed_refusal',
              generation_ref=key('generation',row,fp),input_audit_ref=ik,output_audit_ref=ok if oa else None)
    systems.append(item);blind_entry(row,final,{k:item[k] for k in ('group','policy','state','generation_ref')})
   def counts_full(v):
    out={k:v[k] for k in ('input_block','output_block','exception_input','exception_generation','exception_output','allow')}
    out.update(total=sum(v.values()),exceptions=sum(v[k] for k in ('exception_input','exception_generation','exception_output')),classifier_blocks=v['input_block']+v['output_block'],simulated_generator_invocations=sum(v.values())-v['input_block']-v['exception_input'])
    return out
   agg=counts_full(counts);agg['by_input_label']={l:counts_full(v) for l,v in bylabel.items()};normal=agg['by_input_label']['unharmful'];agg['normal_classifier_block_rate']=normal['classifier_blocks']/normal['total'];agg['normal_not_released_including_exceptions']=(normal['total']-normal['allow'])/normal['total']
   summary['systems'][protectedgroup+'/'+policy]=agg
 sessions=[json.loads(x.read_text()) for x in (LOCAL/'sessions').glob('*.json')]
 summary['sessions']=sessions;summary['disk_after']=resource();summary['protocol_sha256']=p['protocol_sha256']
 summary['actual_new_generations']=sum(s.get('new_count',0) for s in sessions);summary['actual_new_guard_outputs']=sum(s.get('output_new',0) for s in sessions)
 if args.phase=='pilot':
  generations=[s for s in sessions if s.get('phase')=='pilot' and 'model' in s];guards=[s for s in sessions if s.get('phase')=='pilot' and 'output_new' in s]
  estimate=sum(s['generation_wall_seconds']/s['new_count']*1699 for s in generations)+sum(s['wall_seconds']/20*1699 for s in guards)
  summary['estimated_full_compute_seconds_including_pilot']=estimate
  assert len(generations)==2 and len(guards)==1 and all(s.get('complete') for s in sessions)
  assert all(m['completed']==20 and m['failed']==0 and m['unexpected_thinking']==0 for m in summary['models'].values())
  assert all(sum(a[k] for k in ('input_block','output_block','exception_input','exception_generation','exception_output','allow'))==20 for a in summary['systems'].values())
  assert shutil.disk_usage(ROOT).free>2**30 and max(s.get('peak_reserved_bytes',0) for s in sessions)<23*2**30
  summary['synthetic_flow_checks']='test_pipeline covers input_block/output_block/allow, unknown, generation failure and namespace isolation; natural pilot path counts retained without reselection'
  dump(HERE/'pilot_summary.json',summary);print('PILOT_OK estimated total seconds',round(estimate));return
 # Deduplicate by exact request+answer, shuffle before assigning blind IDs; no group/model/Guard fields in blind text file.
 pairs=sorted(blind);random.Random(CFG['blind_seed']).shuffle(pairs);ids={pair:f'blind-{i:05d}' for i,pair in enumerate(pairs)}
 def jsonl(path,items):path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in items))
 jsonl(LOCAL/'blind_review.jsonl',[dict(blind_id=ids[k],**blind[k]) for k in pairs])
 jsonl(LOCAL/'blind_mapping.jsonl',[dict(blind_id=ids[x['content_key']],**x) for x in mapping])
 jsonl(LOCAL/'systems.jsonl',systems)
 summary['blind']={'unique_request_answer_pairs':len(blind),'mapping_rows':len(mapping),'human_labels_filled':0,
                   'files':{name:sha((LOCAL/name).read_bytes()) for name in ['blind_review.jsonl','blind_mapping.jsonl','systems.jsonl']}}
 dump(HERE/'results.json',summary);print('FULL_RECORDS_COMPLETE',len(rows),summary['blind']['unique_request_answer_pairs'])
if __name__=='__main__':main()
