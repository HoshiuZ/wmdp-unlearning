"""No inference: verify exclusive outcomes, every local mapping, blind isolation and public privacy."""
import argparse,re
from collections import Counter
from .common import *
from .policy import outcome

def main():
 p=protocol();r=json.loads((HERE/'results.json').read_text());rows=records();byid={x['sample_id']:x for x in rows};assert len(rows)==1699
 assert sha((LOCAL/'requests.jsonl').read_bytes())==p['request_manifest_sha256']
 assert sha((ROOT/CFG['guard_config']).read_bytes())==p['guard_config_sha256']
 assert sha((ROOT/'safety/qwen3guard/moderator.py').read_bytes())==p['guard_parser_sha256']
 c=db();verified=0
 for model in ('original','rmu80'):
  fp=p['base_fingerprint' if model=='original' else 'rmu80_fingerprint'];good=fail=cap=0
  for row in rows:
   g=get(c,key('generation',row,fp));assert g is not None;verified+=1
   if g['status']=='ok':
    good+=1;cap+=g['hit_generation_limit'];assert sha(g['response'].encode())==g['response_sha256'];assert 0<g['output_tokens']<=512
    assert g['sample_id']==row['sample_id'] and g['model_fingerprint']==fp and not g['truncated']
   else:fail+=1
  assert (good,fail,cap)==tuple(r['models'][model][k] for k in ('completed','failed','hit_generation_limit'))
  sessions=[s for s in r['sessions'] if s.get('model')==model]
  assert all(s['parameters_verified']==398 and s['model_fingerprint']==fp and s.get('complete') for s in sessions)
  assert all(s['generation_config']['do_sample'] is False and s['generation_config']['max_new_tokens']==512 for s in sessions)
 systems=[json.loads(x) for x in (LOCAL/'systems.jsonl').read_text().splitlines()];assert len(systems)==4*1699
 counters={}
 for s in systems:
  row=byid[s['sample_id']];model='original' if s['group']=='B' else 'rmu80';fp=p['base_fingerprint' if model=='original' else 'rmu80_fingerprint']
  a=get(c,s['input_audit_ref']);g=get(c,s['generation_ref']);oa=get(c,s['output_audit_ref']) if s['output_audit_ref'] else None
  assert s['state']==outcome(a,g,oa,s['policy'])
  assert s['actual_simulated_generator_invocation']==(s['state'] not in ('input_block','exception_input'))
  assert s['final_response_ref']==(s['generation_ref'] if s['state']=='allow' else 'config.fixed_refusal')
  counters.setdefault(s['group']+'/'+s['policy'],Counter())[s['state']]+=1
 for name,counter in counters.items():
  for k,v in counter.items():assert r['systems'][name][k]==v
  assert sum(counter.values())==1699
 blind=[json.loads(x) for x in (LOCAL/'blind_review.jsonl').read_text().splitlines()];b={x['blind_id']:x for x in blind};assert len(b)==len(blind)
 assert all(set(x)=={'blind_id','request','answer'} for x in blind)
 mapping=[json.loads(x) for x in (LOCAL/'blind_mapping.jsonl').read_text().splitlines()]
 for m in mapping:
  original=byid[m['sample_id']];answer=get(c,m['generation_ref']).get('response') if m['state'] in ('allow','raw_answer') else CFG['fixed_refusal']
  assert b[m['blind_id']]['request']==original['request'] and b[m['blind_id']]['answer']==answer
  assert digest([original['request'],answer])==m['content_key']
 assert len(mapping)==r['blind']['mapping_rows'];assert len(mapping)==4*1699+sum(m['completed'] for m in r['models'].values())
 for file,hash_ in r['blind']['files'].items():assert sha((LOCAL/file).read_bytes())==hash_
 # Check newly public artifacts against raw prompts, generated replies and raw guard completions.
 texts={x['request'] for x in rows if len(x['request'])>=32}
 for (encoded,) in c.execute('SELECT result FROM records'):
  obj=json.loads(encoded)
  for field in ('response','raw_output'):
   val=obj.get(field)
   if isinstance(val,str) and len(val)>=32:texts.add(val)
 allowed={'.py','.json','.md'};published=[]
 for file in HERE.iterdir():
  if not file.is_file():continue
  assert file.suffix in allowed and file.stat().st_size<2**20
  body=file.read_text();assert not re.search(r'hf_[A-Za-z0-9]{20,}',body)
  # Fixed public refusal is intentionally configured; no original or model-generated texts.
  assert not any(t in body for t in texts if t!=CFG['fixed_refusal'])
  published.append(file.name)
 local_blocks=sum(x.stat().st_blocks*512 for x in LOCAL.rglob('*') if x.is_file())
 out=dict(generation_records_verified=verified,system_rows_verified=len(systems),blind_pairs_verified=len(blind),blind_mapping_rows_verified=len(mapping),
          expected_fingerprints_verified=True,guard_config_parser_unchanged=True,raw_text_and_credentials_not_found_in_public=True,
          public_files_checked=published,local_allocated_bytes=local_blocks,after_resources=resource())
 dump(HERE/'verification.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
