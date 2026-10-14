"""Generate exactly the fixed 40 selected prompts, changing only max_new_tokens."""
import argparse
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from .common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True,choices=['original','rmu80']);a=ap.parse_args();c=connection();selected=rows();pending=[r for r in selected if get(c,key('generate1024',a.model,r)) is None]
 if not pending:print('ALREADY_CACHED',a.model);return
 assert len(selected)==40 and sha((LOCAL/'requests.jsonl').read_bytes())==manifest()['sample_manifest_sha256']
 torch.manual_seed(CFG['seed']);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
 tok=AutoTokenizer.from_pretrained(ROOT/CFG['base_path'],local_files_only=True);tok.padding_side='left';assert sha(tok.chat_template.encode())==old.protocol()['chat_template_sha256']
 model=AutoModelForCausalLM.from_pretrained(ROOT/CFG['base_path'],dtype=torch.bfloat16,device_map=CFG['device'],attn_implementation=CFG['attention'],local_files_only=True);model.eval();model.requires_grad_(False)
 sys.path.insert(0,str(ROOT/'scripts'));from rmu_adaptation_utils import load_delta,tensor_hash
 expected=json.loads((old.LOCAL/'base_fingerprints.json').read_text());assert {n:tensor_hash(v) for n,v in model.named_parameters()}==expected
 if a.model=='rmu80':load_delta(model,ROOT/CFG['delta_path']);expected=json.loads((old.LOCAL/'rmu80_fingerprints.json').read_text())
 assert {n:tensor_hash(v) for n,v in model.named_parameters()}==expected
 assert digest(expected)==manifest()['generator_fingerprints'][a.model]
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 gen=GenerationConfig(**CFG['generation'],eos_token_id=eos,pad_token_id=model.generation_config.pad_token_id,bos_token_id=model.generation_config.bos_token_id)
 session={'model':a.model,'load_verify_seconds':time.perf_counter()-start,'generation_config':gen.to_dict(),'torch':torch.__version__,'transformers':transformers.__version__,'dtype':str(model.dtype),'parameters_verified':len(expected),'model_fingerprint':digest(expected),'source_sha256':sha(Path(__file__).read_bytes()),'batches':[]}
 for offset in range(0,len(pending),20):
  batch=pending[offset:offset+20];texts=[tok.apply_chat_template([{'role':'user','content':r['request']}],tokenize=False,**CFG['chat_template_kwargs']) for r in batch]
  lengths=[len(tok(t,add_special_tokens=False)['input_ids']) for t in texts];assert max(lengths)+1024<=min(model.config.max_position_embeddings,tok.model_max_length),'Input exceeds context; no truncation'
  inputs=tok(texts,return_tensors='pt',padding=True,truncation=False,add_special_tokens=False).to(CFG['device']);width=inputs['input_ids'].shape[-1];torch.cuda.synchronize();bs=time.perf_counter()
  with torch.inference_mode():out=model.generate(**inputs,generation_config=gen)
  torch.cuda.synchronize();seconds=time.perf_counter()-bs
  for j,r in enumerate(batch):
   ids=out[j,width:].tolist();end=next((k+1 for k,v in enumerate(ids) if v in eos),len(ids));ids=ids[:end];ended=bool(ids and ids[-1] in eos);text=tok.decode(ids,skip_special_tokens=True);prior=oldgen(r,a.model)
   assert sha(texts[j].encode())==prior['input_rendered_sha256'] and lengths[j]==prior['input_tokens']
   result={'status':'ok' if text.strip() else 'error','error':None if text.strip() else 'empty_generation','response':text,'response_sha256':sha(text.encode()),'input_tokens':lengths[j],'output_tokens':len(ids),'eos':ended,'finish_reason':'eos' if ended else 'max_new_tokens','cutoff':not ended and len(ids)==1024,'token_ids':ids,'prefix_at_old_token_count':tok.decode(ids[:prior['output_tokens']],skip_special_tokens=True),'input_rendered_sha256':sha(texts[j].encode()),'batch_size':len(batch),'padded_width':width,'truncated':False}
   put(c,key('generate1024',a.model,r),'generate1024',result)
   if not text.strip():raise RuntimeError('Empty generation; saved as error, stop')
  session['batches'].append({'n':len(batch),'seconds':seconds});session.update(wall_seconds=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
  dump(LOCAL/f'{a.model}_session.json',session);print(a.model,'completed',offset+len(batch),flush=True)
 session['complete']=True;dump(LOCAL/f'{a.model}_session.json',session)
if __name__=='__main__':main()
