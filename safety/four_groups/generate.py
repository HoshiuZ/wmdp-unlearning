"""Generate each base/RMU answer once, freeze all weights, never save a full model."""
import argparse,sys,time
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from .common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['original','rmu80'],required=True);ap.add_argument('--phase',choices=['pilot','full'],required=True);args=ap.parse_args()
 torch.manual_seed(CFG['seed']);p=protocol();fingerprint=p['base_fingerprint' if args.model=='original' else 'rmu80_fingerprint'];c=db()
 rows=select(records(),args.phase);pending=[r for r in rows if get(c,key('generation',r,fingerprint)) is None]
 if not pending:print('ALL_CACHED',args.model,args.phase,flush=True);return
 started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 tok=AutoTokenizer.from_pretrained(ROOT/CFG['base_path'],local_files_only=True);tok.padding_side='left'
 assert sha(tok.chat_template.encode())==p['chat_template_sha256']
 model=AutoModelForCausalLM.from_pretrained(ROOT/CFG['base_path'],dtype=torch.bfloat16,device_map=CFG['device'],attn_implementation=CFG['attention'],local_files_only=True)
 model.eval();model.requires_grad_(False)
 sys.path.insert(0,str(ROOT/'scripts'));from rmu_adaptation_utils import tensor_hash,load_delta
 expected=json.loads((LOCAL/'base_fingerprints.json').read_text());actual={n:tensor_hash(v) for n,v in model.named_parameters()};assert actual==expected
 if args.model=='rmu80':
  load_delta(model,ROOT/CFG['delta_path']);expected=json.loads((LOCAL/'rmu80_fingerprints.json').read_text());actual={n:tensor_hash(v) for n,v in model.named_parameters()};assert actual==expected
 assert digest(actual)==fingerprint
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;pad=model.generation_config.pad_token_id
 gen=GenerationConfig(**CFG['generation'],eos_token_id=eos,pad_token_id=pad,bos_token_id=model.generation_config.bos_token_id)
 limit=min(model.config.max_position_embeddings,tok.model_max_length)-CFG['generation']['max_new_tokens']
 assert not model.training and not any(v.requires_grad for v in model.parameters())
 session=dict(model=args.model,phase=args.phase,started_unix=time.time(),model_fingerprint=fingerprint,parameters_verified=len(actual),
              template_sha256=p['chat_template_sha256'],context_input_limit=limit,generation_config=gen.to_dict(),
              python=sys.version,torch=torch.__version__,transformers=transformers.__version__,dtype=str(model.dtype),device=str(model.device),
              sources={f:sha((HERE/f).read_bytes()) for f in ['generate.py','common.py']},new_count=0,new_input_tokens=0,new_output_tokens=0,batches=[])
 sessions=LOCAL/'sessions';sessions.mkdir(exist_ok=True);dest=sessions/f'generate_{args.model}_{args.phase}_{time.time_ns()}.json'
 load_seconds=time.perf_counter()-started;session['load_and_fingerprint_seconds']=load_seconds;runstart=time.perf_counter();streak=0
 for offset in range(0,len(pending),CFG['batch_size']):
  batch=pending[offset:offset+CFG['batch_size']];texts=[tok.apply_chat_template([{'role':'user','content':r['request']}],tokenize=False,**CFG['chat_template_kwargs']) for r in batch]
  lengths=[len(tok(t,add_special_tokens=False)['input_ids']) for t in texts]
  if any(n>limit for n in lengths):
   for r,n in zip(batch,lengths):
    if n>limit:put(c,key('generation',r,fingerprint),'generation',r['sample_id'],dict(status='error',error='input_context_exceeded',model=args.model,input_tokens=n,truncated=False))
   raise RuntimeError('Input context exceeded; no truncation; stop')
  inputs=tok(texts,return_tensors='pt',padding=True,truncation=False,add_special_tokens=False).to(CFG['device']);width=inputs['input_ids'].shape[-1]
  torch.cuda.synchronize();bs=time.perf_counter()
  try:
   with torch.inference_mode():outputs=model.generate(**inputs,generation_config=gen)
   torch.cuda.synchronize();batch_seconds=time.perf_counter()-bs
   for j,(r,n) in enumerate(zip(batch,lengths)):
    ids=outputs[j,width:].tolist();end=next((k+1 for k,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
    ended=bool(ids and ids[-1] in eos);text=tok.decode(ids,skip_special_tokens=True)
    result=dict(status='ok' if text.strip() else 'error',error=None if text.strip() else 'empty_generation',model=args.model,
                response=text,response_sha256=sha(text.encode()),input_tokens=n,output_tokens=len(ids),eos=ended,
                finish_reason='eos' if ended else 'max_new_tokens',hit_generation_limit=not ended and len(ids)>=512,
                input_rendered_sha256=sha(texts[j].encode()),truncated=False,batch_seconds=batch_seconds,batch_size=len(batch),
                contains_think_tags='<think>' in text or '</think>' in text,
                sample_id=r['sample_id'],content_sha256=r['content_sha256'],model_fingerprint=fingerprint)
    put(c,key('generation',r,fingerprint),'generation',r['sample_id'],result)
    session['new_count']+=1;session['new_input_tokens']+=n;session['new_output_tokens']+=len(ids)
    streak=streak+1 if result['status']!='ok' else 0
    if result['contains_think_tags']:raise RuntimeError('Unexpected thinking format in non-thinking model; stop for review')
    if streak>=CFG['max_consecutive_failures']:raise RuntimeError('Sustained generation failures')
   session['batches'].append(dict(n=len(batch),seconds=batch_seconds,input_max_tokens=max(lengths),padded_input_width=width))
  except Exception as exc:
   for r,n in zip(batch,lengths):
    if get(c,key('generation',r,fingerprint)) is None:put(c,key('generation',r,fingerprint),'generation',r['sample_id'],dict(status='error',error=type(exc).__name__,model=args.model,input_tokens=n,truncated=False))
   session.update(error=type(exc).__name__,wall_seconds=time.perf_counter()-started);dump(dest,session);raise
  session.update(generation_wall_seconds=time.perf_counter()-runstart,wall_seconds=time.perf_counter()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),after_resources=resource())
  dump(dest,session);print(json.dumps(dict(model=args.model,phase=args.phase,new=session['new_count'],pending=len(pending),seconds=round(session['generation_wall_seconds'],2))),flush=True)
 session['complete']=True;dump(dest,session)
 print('COMPLETE',args.model,args.phase,flush=True)
if __name__=='__main__':main()
