"""Apply the explicit audited ownership plan; delete only named non-RMU files/OIDs."""
import json,subprocess,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def main():
 plan=json.loads((R/'maintenance/pure_rmu_cleanup_plan.json').read_text());remote=json.loads((R/'maintenance/pure_rmu_remote_lfs_audit.json').read_text());verified={x['oid'] for x in remote['objects'] if x['retrievable']};assert set(plan['delete_lfs'])<=verified
 # The shared fingerprints, fixed evaluation subset and original development data survive elsewhere.
 shared=[('reference/rmu_ce/base_weight_fingerprints.json','open-unlearning/saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json'),('reference/rmu_ce/dev_split.json','logs/rmu_adaptation_dev_split.json'),('reference/rmu_ce/original_development_samples.json','open-unlearning/saves/unlearn/rmu_adaptation_v1/original_development_protocol_check/samples.json')]
 for a,b in shared:assert hashlib.sha256((R/a).read_bytes()).digest()==hashlib.sha256((R/b).read_bytes()).digest()
 tracked=set(plan['delete_files']);unique=[]
 for root in ['open-unlearning/saves/unlearn/rmu_ce_v1','reference/rmu_ce','configs/rmu_ce']:
  for p in (R/root).rglob('*'):
   if p.is_file() and str(p.relative_to(R)) not in tracked and p.suffix in ['.gz','.safetensors','.pt','.pth','.bin']:unique.append({'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'consequence':'Unique untracked candidate weight; do not delete without separate authorization'})
 if unique:
  (R/'maintenance/pure_rmu_unuploaded_weights.json').write_text(json.dumps(unique,indent=2)+'\n');raise RuntimeError('Untracked weight payloads require a separate report and user decision')
 changes=subprocess.check_output(['git','diff','--name-only','--',*plan['delete_files']],cwd=R,text=True);assert not changes.strip(),'Planned deleted files were modified after the starting snapshot'
 before=shutil.disk_usage(R)._asdict();removed=[]
 for s in plan['delete_files']:
  p=R/s;assert p.resolve().is_relative_to(R)
  if p.exists():st=p.stat();removed.append({'path':s,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512});p.unlink()
 # Remove only now-empty method-specific directories; shared roots are never removed.
 for name in ['open-unlearning/saves/unlearn/rmu_ce_v1','reference/rmu_ce','configs/rmu_ce']:
  root=R/name
  for p in sorted((x for x in root.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):
   if not any(p.iterdir()):p.rmdir()
  if root.exists() and not any(root.iterdir()):root.rmdir()
 for s in plan['delete_files']:
  if s.startswith('scripts/') and s.endswith('.py'):
   for p in (R/'scripts/__pycache__').glob(Path(s).stem+'.*.pyc'):p.unlink()
 cache_removed=[]
 for oid in plan['delete_lfs']:
  assert oid not in plan['restore_lfs'];p=R/'.git/lfs/objects'/oid[:2]/oid[2:4]/oid
  if p.exists():
   assert hashlib.sha256(p.read_bytes()).hexdigest()==oid;st=p.stat();cache_removed.append({'oid':oid,'bytes':st.st_size,'allocated_bytes':st.st_blocks*512,'remote_retrieval_verified':True});p.unlink()
 report={'source_commit':plan['source_commit'],'archive_tag':plan['archive_tag'],'removed_files':removed,'removed_lfs_objects':cache_removed,'actual_removed_file_allocated_bytes':sum(x['allocated_bytes'] for x in removed),'actual_removed_lfs_allocated_bytes':sum(x['allocated_bytes'] for x in cache_removed),'space_before_deletion':before,'space_after_deletion':shutil.disk_usage(R)._asdict(),'unuploaded_unique_weights':[],'global_prune_executed':False,'history_rewritten':False,'shared_reference_copies_preserved':shared,'training_started':False,'full_evaluation_started':False}
 (R/'maintenance/pure_rmu_cleanup_execution.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n');print('EXACT_CLEANUP_DONE',len(removed),'files',len(cache_removed),'verified remote LFS objects',flush=True)
if __name__=='__main__':main()
