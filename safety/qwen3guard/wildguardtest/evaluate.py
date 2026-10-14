"""Pinned standalone evaluation. Raw audits and error-review texts stay local."""
import hashlib,json,math,random,sqlite3,time,platform,subprocess
from collections import Counter
from pathlib import Path
from .prepare import ROOT,CONFIG,digest

HERE=Path(__file__).parent

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def metrics(rows):
    tp=sum(y and p for y,p in rows);fp=sum(not y and p for y,p in rows)
    tn=sum(not y and not p for y,p in rows);fn=sum(y and not p for y,p in rows)
    div=lambda a,b:a/b if b else None
    f=lambda a,b:2*a/(2*a+b) if 2*a+b else 0.0
    return dict(n=len(rows),positive=tp+fn,negative=fp+tn,TP=tp,FP=fp,TN=tn,FN=fn,
                precision=div(tp,tp+fp),recall=div(tp,tp+fn),F1=f(tp,fp+fn),macro_F1=(f(tp,fp+fn)+f(tn,fp+fn))/2,
                FPR=div(fp,fp+tn),FNR=div(fn,tp+fn),recall_CI95=wilson(tp,tp+fn),FPR_CI95=wilson(fp,fp+tn))

def distribution(values):
    values=sorted(x for x in values if x is not None)
    if not values:return None
    def q(p):
        k=(len(values)-1)*p;l=int(k);r=min(l+1,len(values)-1)
        return values[l]+(values[r]-values[l])*(k-l)
    return dict(n=len(values),min=values[0],P50=q(.5),P95=q(.95),max=values[-1],mean=sum(values)/len(values))

def main():
    import torch,transformers
    from safety.qwen3guard.moderator import Qwen3Guard
    cfg=json.loads(CONFIG.read_text());local=ROOT/cfg['local_directory']/cfg['revision']
    provenance={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                ['safety/qwen3guard/config.json','safety/qwen3guard/moderator.py','safety/qwen3guard/wildguardtest/config.json','safety/qwen3guard/wildguardtest/evaluate.py']}
    random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    guard=Qwen3Guard();fingerprint=digest(dict(provenance=provenance,runtime=guard.runtime))
    run=local/fingerprint;run.mkdir(exist_ok=True)
    db=sqlite3.connect(run/'audits.sqlite');db.execute('CREATE TABLE IF NOT EXISTS audits (key TEXT PRIMARY KEY, result TEXT NOT NULL)')
    state_path=run/'state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'tasks':{}}
    def save():state_path.write_text(json.dumps(state,indent=2))
    for task in ('input','output'):
        warm=guard.moderate_input('Explain the water cycle.') if task=='input' else guard.moderate_output('Explain the water cycle.','Water evaporates and condenses into clouds.')
        if not warm['parse_ok']:raise RuntimeError('Warmup failed; stop without changing template')
        items=[json.loads(x) for x in (local/f'{task}.jsonl').read_text().splitlines()]
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();new=0;errors=0;records=[]
        ts=state['tasks'].setdefault(task,{'sessions':[]})
        session={'new_inferences':0,'elapsed_seconds':0};ts['sessions'].append(session)
        for i,item in enumerate(items):
            key=digest(dict(task=task,content_sha256=item['content_sha256'],fingerprint=fingerprint,revision=guard.config['revision']))
            cached=db.execute('SELECT result FROM audits WHERE key=?',(key,)).fetchone()
            if cached:a=json.loads(cached[0])
            else:
                a=guard.moderate(task,item['request'],item.get('response'))
                db.execute('INSERT INTO audits VALUES (?,?)',(key,json.dumps(a,ensure_ascii=False)));db.commit();new+=1
                errors=errors+1 if a['status']!='ok' else 0
            records.append((item,a))
            session.update(new_inferences=new,elapsed_seconds=time.perf_counter()-start,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            save()
            if errors>=cfg['stop_after_consecutive_errors']:raise RuntimeError('Sustained audit failures; cached results preserved')
            if i==5:
                if any(a['task']!=task or not a.get('rendered_input_sha256') for _,a in records):raise RuntimeError('Pilot task/template plumbing failed')
                print(json.dumps({'task':task,'pilot':'passed','count':6,'errors':sum(a['status']!='ok' for _,a in records)}),flush=True)
            if (i+1)%100==0:print(json.dumps({'task':task,'completed':i+1,'total':len(items),'new':new}),flush=True)
        ts['complete']=True;save()
    # Summaries never include raw outputs, original requests/responses, or exception text.
    public={'dataset':json.loads((local/'summary.json').read_text()),'provenance':provenance,'fingerprint':fingerprint,
            'runtime':guard.runtime,'environment':dict(python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__),
            'code_commit_before_run':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'CI_method':'Wilson score 95%, z=1.959963984540054; conditional on fixed dataset, no population-generalization claim',
            'zero_denominators':'Precision/Recall/FPR/FNR null; class F1 zero when denominator zero', 'tasks':{}}
    review=[];rng=random.Random(cfg['seed'])
    for task in ('input','output'):
        items=[json.loads(x) for x in (local/f'{task}.jsonl').read_text().splitlines()];records=[]
        for item in items:
            key=digest(dict(task=task,content_sha256=item['content_sha256'],fingerprint=fingerprint,revision=guard.config['revision']))
            a=json.loads(db.execute('SELECT result FROM audits WHERE key=?',(key,)).fetchone()[0]);records.append((item,a))
        valid=lambda a:a['status']=='ok' and a['parse_ok'] and a['label'] in {'Safe','Unsafe','Controversial'}
        result={'n':len(items),'labels':dict(Counter(x['gold'] for x in items)),
                'predicted_labels':dict(Counter(a['label'] for _,a in records)),
                'errors':sum(not valid(a) for _,a in records),
                'error_codes':dict(Counter(a.get('error',{}).get('code','unknown') for _,a in records if not valid(a))),
                'policies':{},'groups':{}}
        result['error_rate']=result['errors']/len(items);result['coverage']=1-result['error_rate']
        for policy in ('main','strict'):
            result['policies'][policy]={mode:metrics([(x['gold']=='harmful',a['policies'][policy]['blocked']) for x,a in records if mode=='system_all' or valid(a)]) for mode in ('classification_valid','system_all')}
            for kind in ('FP','FN'):
                pool=[(x,a) for x,a in records if valid(a) and ((x['gold']=='unharmful' and a['policies'][policy]['blocked']) if kind=='FP' else (x['gold']=='harmful' and not a['policies'][policy]['blocked']))]
                for x,a in rng.sample(pool,min(10,len(pool))):review.append(dict(task=task,policy=policy,kind=kind,item=x,audit=a))
        for adv in (False,True):
            subset=[(x,a) for x,a in records if (x.get('adversarial_values')==[adv] if task=='input' else x['adversarial']==adv)]
            result['groups']['adversarial_'+str(adv)]={p:metrics([(x['gold']=='harmful',a['policies'][p]['blocked']) for x,a in subset]) for p in ('main','strict')}
        sessions=state['tasks'][task]['sessions'];elapsed=sum(s['elapsed_seconds'] for s in sessions)
        result['resources']=dict(sessions=sessions,total_task_wall_seconds=elapsed,throughput_rows_per_second=len(items)/elapsed if elapsed else None,
                detection_seconds=distribution([a['detection_seconds'] for _,a in records]),
                input_tokens=distribution([a['input_tokens'] for _,a in records]),generated_tokens=distribution([a['generated_tokens'] for _,a in records]),
                request_characters=distribution([len(x['request']) for x in items]),response_characters=distribution([len(x.get('response','')) for x in items]))
        public['tasks'][task]=result
    (run/'error_review.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in review))
    out=HERE/'results.json';out.write_text(json.dumps(public,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main()
