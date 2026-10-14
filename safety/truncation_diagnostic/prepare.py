import random
from collections import Counter
from .common import *

def main():
 if (HERE/'REPORT.md').exists() and (HERE/'results.json').exists():print('ALREADY_COMPLETED; do not repeat inference');return
 original=old.records();p=old.protocol();counts={};eligible=[]
 for model in ('original','rmu80'):
  gs=[oldgen(r,model) for r in original]
  assert all(g['status']=='ok' and g['finish_reason']==('eos' if g['eos'] else 'max_new_tokens') for g in gs)
  counts[model]={'n':len(gs),'natural':sum(g['eos'] for g in gs),'cutoff':sum(not g['eos'] and g['output_tokens']==512 for g in gs),'natural_at_512':sum(g['eos'] and g['output_tokens']==512 for g in gs)}
 for r in original:
  if any(not (g:=oldgen(r,m))['eos'] and g['output_tokens']==512 for m in ('original','rmu80')):eligible.append(r)
 rng=random.Random(42);ids=[]
 for label in ('harmful','unharmful'):
  pool=[r for r in eligible if r['gold']==label];assert len(pool)>=20;ids += [r['sample_id'] for r in rng.sample(pool,20)]
 selected=[r for r in original if r['sample_id'] in ids];assert len(selected)==40
 text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected);target=LOCAL/'requests.jsonl'
 if target.exists():assert target.read_text()==text
 else:target.write_text(text)
 base=copy.deepcopy(CFG);base['generation']['max_new_tokens']=512;assert base==old.CFG
 out={'seed':42,'selection':'Union of actual non-EOS length-512 endings in either model; 20 harmful +20 unharmful, random.sample per label then restore source order',
      'eligible_n':len(eligible),'eligible_labels':dict(Counter(r['gold'] for r in eligible)),'sample_n':40,'sample_ids':ids,
      'ordered_sample_ids':[r['sample_id'] for r in selected],'sample_manifest_sha256':sha(target.read_bytes()),'original_endings':counts,
      'original_protocol_sha256':p['protocol_sha256'],'config':CFG,'only_generation_setting_changed':'max_new_tokens 512 ->1024',
      'source_before_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'generator_fingerprints':{m:p['base_fingerprint' if m=='original' else 'rmu80_fingerprint'] for m in ('original','rmu80')},
      'guard_revision':p['guard_revision'],'guard_config_sha256':p['guard_config_sha256'],'guard_parser_sha256':p['guard_parser_sha256'],
      'prefix_caveat':'Batch size stays20; the selected40 create different co-batches from the historical1699. BF16 greedy numerics may change prefixes; mismatch is not attributed solely to max_new_tokens.',
      'before_resources':resource()}
 out['sha256']=digest({k:v for k,v in out.items() if k not in ('before_resources','source_before_commit')})
 if (HERE/'manifest.json').exists():assert manifest()['sha256']==out['sha256']
 else:dump(HERE/'manifest.json',out)
 print('ORIGINAL_ENDINGS',counts,'ELIGIBLE',len(eligible),'SELECTED',len(selected))
if __name__=='__main__':main()
