"""One fixed18-case interface check. Never invokes the original or RMU model."""
import argparse,collections,hashlib,json,platform,shutil,statistics,subprocess,sys,time
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import torch,transformers
from .moderator import ROOT,Qwen3Guard
HERE=Path(__file__).resolve().parent

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    values=sorted(values);pos=(len(values)-1)*.95;lo=int(pos);hi=min(lo+1,len(values)-1)
    return {'n':len(values),'mean_seconds':statistics.mean(values),'median_seconds':statistics.median(values),'p95_seconds':values[lo]+(values[hi]-values[lo])*(pos-lo),'min_seconds':values[0],'max_seconds':values[-1]}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(HERE/'records/smoke_v1'));args=parser.parse_args();out=Path(args.output).resolve()
    if out.exists():raise RuntimeError('Refuse to overwrite an existing smoke run')
    out.mkdir(parents=True);before=shutil.disk_usage(ROOT)._asdict();start=time.perf_counter()
    env={'recorded_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,'transformers':transformers.__version__,'cuda_runtime':torch.version.cuda,'gpu_before':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,memory.used,driver_version','--format=csv'],text=True).strip(),'space_before':before,'code_commit_at_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'moderator.py',HERE/'run_smoke.py',HERE/'config.json',HERE/'samples.jsonl']},'command':'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 envs/unlearning/bin/python -m safety.qwen3guard.run_smoke','scope':'Only Guard and18 authored smoke cases; no training, no A/B/C/D formal evaluation, no original/RMU generation'}
    dump(out/'environment.json',env)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.cuda.reset_peak_memory_stats()
    loadstart=time.perf_counter();guard=Qwen3Guard();runtime=dict(guard.runtime,load_seconds=time.perf_counter()-loadstart,seed=0);dump(out/'runtime.json',runtime)
    warmups=[guard.moderate_input('你好，请介绍一下自己。'),guard.moderate_output('What is two plus two?','Two plus two is four.')]
    dump(out/'warmup.json',{'excluded_from_latency_statistics':True,'results':warmups})
    samples=[json.loads(l) for l in (HERE/'samples.jsonl').open()];assert len(samples)==18
    records=[]
    with (out/'results.jsonl').open('w') as stream:
        for case in samples:
            result=guard.moderate(case['task'],case['request'],case['response'])
            row=dict(case,audit=result,qualitative_flags=[])
            if result['parse_ok']:
                if result['label'] not in case['informal_expected_labels']:row['qualitative_flags'].append('informal_label_expectation_mismatch')
                if case['informal_expected_refusal'] is not None and result['refusal']!=case['informal_expected_refusal']:row['qualitative_flags'].append('informal_refusal_expectation_mismatch')
            else:row['qualitative_flags'].append('audit_error_or_unknown')
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();records.append(row)
            print(case['id'],result['label'],result['categories'],result['refusal'],round(result['detection_seconds'],4),row['qualitative_flags'],flush=True)
    calls_before=getattr(guard,'generation_calls',0)
    overflow=guard.moderate_input('hello '*33000)
    calls_after=getattr(guard,'generation_calls',0)
    assert overflow['error']['code']=='context_length_exceeded' and calls_before==calls_after
    dump(out/'overflow_check.json',{'probe':'33000 repetitions of benign hello; body not stored','audit':overflow,'model_generation_calls_added':calls_after-calls_before})
    assert calls_after==len(samples)+len(warmups),'Two policy mappings must not duplicate inference'
    counts={task:dict(collections.Counter(r['audit']['label'] for r in records if r['task']==task)) for task in ['input','output']}
    policies={name:{'risk_label_blocks':sum(r['audit']['policies'][name]['risk_label_block'] for r in records),'error_unknown_holds':sum(r['audit']['policies'][name]['error_unknown'] for r in records),'allow':sum(r['audit']['policies'][name]['action']=='allow' for r in records)} for name in guard.config['policies']}
    summary={'status':'completed','samples':18,'warmup_samples':len(warmups),'model_generation_calls':calls_after,'overflow_probe_generation_calls':0,'label_counts_by_task':counts,'parse_not_success_count':sum(not r['audit']['parse_ok'] for r in records),'parse_error_count':sum(r['audit'].get('error',{}).get('code')=='parse_error' for r in records),'generation_error_count':sum(r['audit'].get('error',{}).get('code') not in [None,'parse_error'] for r in records),'policy_counts':policies,'qualitative_flags':{r['id']:r['qualitative_flags'] for r in records if r['qualitative_flags']},'latency_after_warmup':{t:stats([r['audit']['detection_seconds'] for r in records if t=='all' or r['task']==t]) for t in ['all','input','output']},'timing_scope':'Synchronous wall time per single audit, template/tokenize/device transfer/generate/decode/parse/two policy maps; excludes model loading and two warmups; no batching or repetitions. generation_seconds separately measures generate plus CUDA synchronization. Overflow probe excluded.','cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'total_process_seconds':time.perf_counter()-start,'space_after':shutil.disk_usage(ROOT)._asdict(),'not_a_safety_accuracy_benchmark':True,'stop_after_this_stage':True}
    dump(out/'summary.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
