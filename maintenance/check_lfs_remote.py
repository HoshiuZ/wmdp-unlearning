"""Read-only remote LFS retrieval audit. Credentials remain in memory and are never logged."""
import json,re,subprocess,urllib.request,concurrent.futures,time,os,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 plan=json.loads((ROOT/'maintenance/pure_rmu_cleanup_plan.json').read_text());url=subprocess.check_output(['git','remote','get-url','origin'],cwd=ROOT,text=True).strip();m=re.fullmatch(r'git@([^:]+):(.+)',url);assert m
 transport=os.environ.get('GIT_SSH_COMMAND') or subprocess.run(['git','config','--get','core.sshCommand'],cwd=ROOT,capture_output=True,text=True).stdout.strip() or 'ssh';argv=[os.path.expanduser(x) for x in shlex.split(transport)];raw=subprocess.check_output(argv+['-o','BatchMode=yes','-o','ConnectTimeout=20','git@'+m[1],'git-lfs-authenticate',m[2],'download'],stderr=subprocess.PIPE);auth=json.loads(raw)
 objects={**plan['restore_lfs'],**plan['delete_lfs']};headers=dict(auth.get('header',{}));headers.update({'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'});items=list(objects.items());batch={'objects':[]}
 for offset in range(0,len(items),20):
  body=json.dumps({'operation':'download','transfers':['basic'],'objects':[{'oid':oid,'size':d['size']} for oid,d in items[offset:offset+20]]}).encode();req=urllib.request.Request(auth['href'].rstrip('/')+'/objects/batch',data=body,headers=headers)
  with urllib.request.urlopen(req,timeout=60) as response:batch['objects'].extend(json.load(response)['objects'])
 def check(obj):
  oid=obj['oid'];result={'oid':oid,'expected_bytes':objects[oid]['size'],'paths':objects[oid]['paths']}
  if 'error' in obj:return dict(result,retrievable=False,error_code=obj['error'].get('code'))
  action=obj.get('actions',{}).get('download')
  if not action:return dict(result,retrievable=False,error_code='no_download_action')
  for attempt in range(3):
   try:
    h=dict(action.get('header',{}));h['Range']='bytes=0-0';r=urllib.request.Request(action['href'],headers=h)
    with urllib.request.urlopen(r,timeout=45) as response:
     data=response.read(1);status=response.status;cr=response.headers.get('Content-Range');length=response.headers.get('Content-Length');size=int(cr.rsplit('/',1)[1]) if cr else int(length)
    assert len(data)==1 and status in [200,206] and size==objects[oid]['size'];return dict(result,retrievable=True,http_status=status,remote_size=size,first_byte_hex=data.hex())
   except Exception as exc:
    if attempt==2:return dict(result,retrievable=False,error_type=type(exc).__name__)
    time.sleep(1)
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(check,batch['objects']))
 assert {r['oid'] for r in rows}==set(objects);out={'operation':'download batch plus actual ranged GET (not upload)','credentials_logged':False,'objects':rows,'all_retrievable':all(r['retrievable'] for r in rows)};(ROOT/'maintenance/pure_rmu_remote_lfs_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('REMOTE_LFS_RETRIEVAL_AUDIT',len(rows),sum(r['retrievable'] for r in rows),flush=True)
 if not out['all_retrievable']:raise RuntimeError('Some objects are not verified; do not delete their local payloads')
if __name__=='__main__':main()
