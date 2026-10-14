"""Check completed metrics/cache consistency and staged artifacts without printing test text."""
import hashlib,json,re,sqlite3,subprocess
from pathlib import Path
import pyarrow.parquet as pq
from .prepare import ROOT,digest
from .evaluate import metrics

HERE=Path(__file__).parent

def main():
    result=json.loads((HERE/'results.json').read_text());cfg=result['dataset']['dataset']
    local=ROOT/cfg['local_directory']/cfg['revision'];fp=result['fingerprint']
    db=sqlite3.connect(local/fp/'audits.sqlite')
    for file,sha in result['provenance'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha, 'Execution source/config changed'
    for task,t in result['tasks'].items():
        items=[json.loads(x) for x in (local/f'{task}.jsonl').read_text().splitlines()];records=[]
        for x in items:
            key=digest(dict(task=task,content_sha256=x['content_sha256'],fingerprint=fp,revision=result['runtime']['revision']))
            cached=db.execute('SELECT result FROM audits WHERE key=?',(key,)).fetchone()
            assert cached is not None, 'Missing persisted audit'
            a=json.loads(cached[0]);assert a['task']==task
            records.append((x,a))
        assert len(records)==t['n']
        assert sum(a['status']!='ok' for _,a in records)==t['errors']
        for policy in ('main','strict'):
            actual=metrics([(x['gold']=='harmful',a['policies'][policy]['blocked']) for x,a in records])
            assert actual==t['policies'][policy]['system_all'], 'Published metrics differ from cache'
    staged=subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode().split('\0')
    staged=[x for x in staged if x]
    assert staged, 'Stage intended public artifacts before running this check'
    raw=ROOT/'cache/huggingface/hub/datasets--allenai--wildguardmix/snapshots'/cfg['revision']/cfg['file']
    table=pq.read_table(raw);texts={x for field in ('prompt','response') for x in table[field].to_pylist() if isinstance(x,str) and len(x)>=32}
    for name in staged:
        assert name=='README.md' or name.startswith('safety/qwen3guard/wildguardtest/'), 'Unrelated staged artifact'
        p=ROOT/name;assert p.suffix in {'.md','.py','.json'}, 'Unexpected file type'
        assert p.stat().st_size<1024*1024, 'Unexpected large file'
        body=p.read_text()
        assert not re.search(r'hf_[A-Za-z0-9]{20,}',body), 'Possible credential'
        assert not any(x in body for x in texts), 'Original test text in public artifact'
    for name in ('results.json','error_summary.json'):
        body=(HERE/name).read_text()
        for key in ('raw_output','raw_fields','rendered_input','other_output_lines'):
            assert '"'+key+'"' not in body, 'Raw audit field in public JSON'
    print(json.dumps({'verified_cached_audits':sum(t['n'] for t in result['tasks'].values()),
                      'verified_staged_files':len(staged),'source_fingerprints_match':True,
                      'raw_test_text_or_token_found':False,'unrelated_or_large_artifacts':False},indent=2))

if __name__=='__main__':main()
