"""Local-only caches; public manifests contain hashes and counts, never text."""
import hashlib,json,sqlite3,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
CFG=json.loads((HERE/'config.json').read_text())
LOCAL=ROOT/CFG['local_root'];LOCAL.mkdir(parents=True,exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def db():
 c=sqlite3.connect(LOCAL/'records.sqlite');c.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, task TEXT, sample TEXT, result TEXT NOT NULL)');return c
def put(c,key,task,sample,value):
 c.execute('INSERT INTO records VALUES (?,?,?,?)',(key,task,sample,json.dumps(value,ensure_ascii=False)));c.commit()
def get(c,key):
 r=c.execute('SELECT result FROM records WHERE key=?',(key,)).fetchone();return json.loads(r[0]) if r else None
def records():return [json.loads(x) for x in (LOCAL/'requests.jsonl').read_text().splitlines()]
def protocol():return json.loads((HERE/'manifest.json').read_text())
def key(task,row,model,content=None):return digest(dict(task=task,content=content or row['content_sha256'],model=model,protocol=protocol()['protocol_sha256'],guard_revision=json.loads((ROOT/CFG['guard_config']).read_text())['revision']))
def select(rows,phase):
 pilot=set(protocol()['pilot_sample_ids']);return [r for r in rows if phase=='full' or r['sample_id'] in pilot]
def resource():
 import shutil
 return dict(disk_free_bytes=shutil.disk_usage(ROOT).free)
