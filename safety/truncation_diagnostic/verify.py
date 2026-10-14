"""Verify immutable source settings, fixed cohort, caches and public privacy without inference."""
import re
from transformers import AutoTokenizer
from .common import *

def main():
 m=manifest();c=connection();sample=rows();assert len(sample)==40 and len({r['sample_id'] for r in sample})==40
 assert sum(r['gold']=='harmful' for r in sample)==20
 assert sha((LOCAL/'requests.jsonl').read_bytes())==m['sample_manifest_sha256']
 for row in sample:assert any(not (g:=oldgen(row,model))['eos'] and g['output_tokens']==512 for model in ('original','rmu80'))
 assert sha((ROOT/CFG['guard_config']).read_bytes())==m['guard_config_sha256'] and sha((ROOT/'safety/qwen3guard/moderator.py').read_bytes())==m['guard_parser_sha256']
 tok=AutoTokenizer.from_pretrained(old.protocol()['guard_runtime']['snapshot_path'],local_files_only=True)
 for model in ('original','rmu80'):
  session=json.loads((LOCAL/f'{model}_session.json').read_text());assert session['complete'] and session['parameters_verified']==398
  assert session['model_fingerprint']==m['generator_fingerprints'][model]
  prior=next(s for s in json.loads((old.HERE/'results.json').read_text())['sessions'] if s.get('model')==model and s['phase']=='full')
  diffs={k:(prior['generation_config'].get(k),v) for k,v in session['generation_config'].items() if prior['generation_config'].get(k)!=v}
  assert diffs=={'max_new_tokens':(512,1024)}
  assert session['source_sha256']==sha((HERE/'generate.py').read_bytes())
  for row in sample:
   oldg=oldgen(row,model);new=get(c,key('generate1024',model,row));assert new['status']=='ok'
   assert sha(oldg['response'].encode())==oldg['response_sha256'];assert sha(new['response'].encode())==new['response_sha256']
   assert oldg['input_rendered_sha256']==new['input_rendered_sha256'] and oldg['input_tokens']==new['input_tokens']
   assert new['eos']==(new['token_ids'][-1] in session['generation_config']['eos_token_id'])
   assert new['cutoff']==(not new['eos'] and new['output_tokens']==1024)
   for n,g in [(512,oldg),(1024,new)]:
    a=get(c,key('audit'+str(n),model,row));assert a is not None
    text=tok.apply_chat_template([{'role':'user','content':row['request']},{'role':'assistant','content':g['response']}],tokenize=False)
    assert sha(text.encode())==a['rendered_input_sha256']
 texts=[r['request'] for r in sample]
 for (value,) in c.execute('select result from records'):
  v=json.loads(value)
  for name in ('response','raw_output'):
   if isinstance(v.get(name),str):texts.append(v[name])
 for file in HERE.iterdir():
  if not file.is_file():continue
  assert file.suffix in {'.py','.json','.md'} and file.stat().st_size<2**20
  text=file.read_text();assert not re.search(r'hf_[A-Za-z0-9]{20,}',text)
  assert not any(v in text for v in texts if len(v)>=32 and v!=CFG['fixed_refusal'])
 out={'sample_n':40,'generation_records':80,'paired_output_audits':160,'both_full_parameter_fingerprints_verified':True,'only_generation_config_change':'max_new_tokens','original_results_read_only':True,'public_sensitive_text_or_tokens_found':False,'disk_free_bytes':resource()['disk_free_bytes']}
 dump(HERE/'verification.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
