"""Fetch only the pinned Guard snapshot; reuse one ignored HF blob cache."""
import json,time,shutil,hashlib
from pathlib import Path
from huggingface_hub import snapshot_download
def main():
    root=Path(__file__).resolve().parents[2];out=root/'safety/qwen3guard/records';source=json.loads((out/'model_source.json').read_text());config=json.loads((root/'safety/qwen3guard/config.json').read_text());assert source['revision']==config['revision'] and source['repo_id']==config['model_id'];start=time.perf_counter();before=shutil.disk_usage(root)._asdict()
    allow=['README.md','LICENSE','config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors','model.safetensors.index.json','model-*.safetensors','chat_template.jinja','special_tokens_map.json','added_tokens.json']
    expected=sum(f['size'] or 0 for f in source['files'] if __import__('fnmatch').fnmatch(f['name'],'*.safetensors'));assert expected<3*2**30,expected
    assert before['free']>expected+2*2**30
    p=Path(snapshot_download(source['repo_id'],revision=source['revision'],cache_dir=str(root/'cache/huggingface/hub'),allow_patterns=allow,max_workers=2))
    files=[]
    for f in sorted(p.iterdir()):
     if not f.is_file():continue
     h=hashlib.sha256()
     with f.open('rb') as stream:
      for b in iter(lambda:stream.read(8*2**20),b''):h.update(b)
     files.append({'name':f.name,'bytes':f.stat().st_size,'sha256':h.hexdigest(),'is_symlink':f.is_symlink()})
    record={'repo_id':source['repo_id'],'revision':source['revision'],'snapshot_path':str(p),'cache_dir':str(root/'cache/huggingface/hub'),'files':files,'total_file_bytes':sum(f['bytes'] for f in files),'elapsed_seconds':time.perf_counter()-start,'space_before':before,'space_after':shutil.disk_usage(root)._asdict(),'no_local_dir_model_copy':True}
    manifest=out/'download_manifest.json'
    if manifest.exists():manifest=out/f'download_recheck_{time.time_ns()}.json'
    manifest.write_text(json.dumps(record,indent=2)+'\n');print('DOWNLOAD_COMPLETE',p,record['total_file_bytes'])

if __name__=="__main__":main()
