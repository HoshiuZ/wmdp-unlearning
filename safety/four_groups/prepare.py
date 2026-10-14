import json,random,subprocess
from pathlib import Path
from transformers import AutoTokenizer
from .common import *

def main():
 source=json.loads((ROOT/'safety/qwen3guard/wildguardtest/results.json').read_text());d=source['dataset'];revision=d['dataset']['revision']
 path=ROOT/'cache/wildguardtest'/revision/'input.jsonl';expected=next(x for x in d['manifests'] if x['file']=='input')
 assert sha(path.read_bytes())==expected['sha256']
 rows=[json.loads(x) for x in path.read_text().splitlines()];assert len(rows)==1699 and len({r['request'] for r in rows})==1699
 assert sum(r['gold']=='harmful' for r in rows)==754 and sum(r['gold']=='unharmful' for r in rows)==945
 for i,r in enumerate(rows):r['sample_id']=f'wg-{i:04d}-'+r['content_sha256'][:12]
 target=LOCAL/'requests.jsonl';body=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
 if target.exists():assert target.read_text()==body
 else:target.write_text(body)
 rng=random.Random(CFG['seed']);pilot=[]
 for label in ('harmful','unharmful'):pilot.extend(rng.sample([r['sample_id'] for r in rows if r['gold']==label],CFG['pilot_n_per_input_label']))
 pilot=[r['sample_id'] for r in rows if r['sample_id'] in pilot]
 base=ROOT/CFG['base_path'];filechecks=[]
 for f in json.loads((ROOT/CFG['base_manifest']).read_text())['files']:
  p=ROOT/f['path'];assert sha(p.read_bytes())==f['sha256'],f['path'];filechecks.append(dict(path=f['path'],sha256=f['sha256']))
 tok=AutoTokenizer.from_pretrained(base,local_files_only=True);template_sha=sha(tok.chat_template.encode())
 delta=json.loads((ROOT/CFG['delta_path']/'checkpoint.json').read_text())
 basefp=json.loads((ROOT/CFG['base_parameter_fingerprints']).read_text());rmufp=dict(basefp)
 for p in delta['parameters']:assert basefp[p['name']]==p['base_tensor_sha256'];rmufp[p['name']]=p['trained_tensor_sha256']
 guard=json.loads((ROOT/CFG['guard_config']).read_text())
 prov=dict(config=CFG,request_manifest_sha256=sha(target.read_bytes()),source_input_manifest_sha256=expected['sha256'],
           base_fingerprint=digest(basefp),rmu80_fingerprint=digest(rmufp),delta_manifest_sha256=sha((ROOT/CFG['delta_path']/'checkpoint.json').read_bytes()),
           base_files=filechecks,chat_template_sha256=template_sha,guard_config_sha256=sha((ROOT/CFG['guard_config']).read_bytes()),guard_parser_sha256=sha((ROOT/'safety/qwen3guard/moderator.py').read_bytes()),
           guard_revision=guard['revision'],guard_runtime=source['runtime'],prior_cache_fingerprint=source['fingerprint'],
           dataset_revision=revision,pilot_sample_ids=pilot,source_n=1699,input_label_counts={'harmful':754,'unharmful':945},
           code_commit_before_run=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),before_resources=resource())
 prov['protocol_sha256']=digest({k:v for k,v in prov.items() if k not in ('before_resources','code_commit_before_run')})
 out=HERE/'manifest.json'
 if out.exists():assert json.loads(out.read_text())['protocol_sha256']==prov['protocol_sha256']
 else:dump(out,prov)
 dump(LOCAL/'base_fingerprints.json',basefp);dump(LOCAL/'rmu80_fingerprints.json',rmufp)
 print('PINNED',len(rows),'pilot',len(pilot),'protocol',prov['protocol_sha256'])
if __name__=='__main__':main()
