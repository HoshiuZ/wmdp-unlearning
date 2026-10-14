"""Independent diagnostic cache; existing four-group records stay read-only."""
import copy,json,sqlite3,time,sys
from pathlib import Path
from safety.four_groups import common as old
ROOT=old.ROOT;HERE=Path(__file__).parent;LOCAL=ROOT/'cache/truncation_diagnostic_v1'
LOCAL.mkdir(parents=True,exist_ok=True)
CFG=copy.deepcopy(old.CFG);CFG['generation']['max_new_tokens']=1024
sha=old.sha;digest=old.digest;dump=old.dump;resource=old.resource

def connection():
 c=sqlite3.connect(LOCAL/'records.sqlite');c.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, task TEXT, result TEXT)');return c

def manifest():return json.loads((HERE/'manifest.json').read_text())
def rows():return [json.loads(x) for x in (LOCAL/'requests.jsonl').read_text().splitlines()]
def oldgen(row,model):
 p=old.protocol();fp=p['base_fingerprint' if model=='original' else 'rmu80_fingerprint']
 c=sqlite3.connect('file:'+str(old.LOCAL/'records.sqlite')+'?mode=ro',uri=True)
 result=old.get(c,old.key('generation',row,fp));c.close();return result

def key(task,model,row,content=None):return digest(dict(task=task,model=model,content=content or row['content_sha256'],protocol=manifest()['sha256']))
def get(c,k):
 v=c.execute('select result from records where key=?',(k,)).fetchone();return json.loads(v[0]) if v else None

def put(c,k,task,v):c.execute('insert into records values (?,?,?)',(k,task,json.dumps(v,ensure_ascii=False)));c.commit()
