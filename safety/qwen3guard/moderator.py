"""Pinned, offline Qwen3Guard adapter. Templates come only from its tokenizer."""
from __future__ import annotations
import hashlib,json,re,time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[2]
LABELS={'Safe','Controversial','Unsafe'}
CATEGORIES={'Violent','Non-violent Illegal Acts','Sexual Content or Sexual Acts','PII','Suicide & Self-Harm','Unethical Acts','Politically Sensitive Topics','Copyright Violation'}

def parse_output(raw: str,task: str) -> dict[str,Any]:
    """Require complete official structured fields; never infer Safe on failure."""
    if task not in {'input','output'}:raise ValueError('task must be input or output')
    fields={};errors=[];other=[]
    for line in raw.splitlines():
        line=line.strip()
        if not line:continue
        match=re.fullmatch(r'([A-Za-z]+):\s*(.*)',line)
        if not match:other.append(line);errors.append('unexpected_text');continue
        key,value=match.groups()
        if key in fields:errors.append('duplicate_'+key)
        fields[key]=value
    expected={'Safety','Categories'}|({'Refusal'} if task=='output' else set())
    for key in sorted(expected-set(fields)):errors.append('missing_'+key)
    for key in sorted(set(fields)-expected):errors.append('unexpected_'+key)
    parsed_label=fields.get('Safety')
    if parsed_label not in LABELS:errors.append('invalid_safety')
    categories=[s.strip() for s in fields.get('Categories','').split(',') if s.strip()]
    allowed=CATEGORIES|({'Jailbreak'} if task=='input' else set())|{'None'}
    if not categories or any(s not in allowed for s in categories):errors.append('invalid_categories')
    if 'None' in categories and categories!=['None']:errors.append('mixed_none_categories')
    if len(categories)!=len(set(categories)):errors.append('duplicate_categories')
    if parsed_label=='Safe' and categories!=['None']:errors.append('safe_category_conflict')
    refusal=fields.get('Refusal')
    if task=='output' and refusal not in {'Yes','No'}:errors.append('invalid_refusal')
    ok=not errors
    return {'status':'ok' if ok else 'error','parse_ok':ok,'label':parsed_label if ok else 'unknown','parsed_label':parsed_label,'categories':categories,'refusal':refusal,'raw_output':raw,'raw_fields':fields,'other_output_lines':other,'parse_errors':errors}

def apply_policy(audit: dict[str,Any],blocked_labels: list[str]) -> dict[str,Any]:
    if not blocked_labels or not set(blocked_labels)<=LABELS:raise ValueError('Invalid policy labels')
    unknown=audit.get('status')!='ok' or not audit.get('parse_ok') or audit.get('label') not in LABELS
    risk_block=not unknown and audit['label'] in blocked_labels
    return {'blocked':bool(unknown or risk_block),'action':'hold_for_review' if unknown else ('block' if risk_block else 'allow'),'reason':'audit_error_or_unknown' if unknown else ('blocked_label' if risk_block else 'label_allowed'),'risk_label_block':risk_block,'error_unknown':unknown}

class Qwen3Guard:
    def __init__(self,config_path: str|Path|None=None):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.config=json.loads(Path(config_path or Path(__file__).with_name('config.json')).read_text())
        cfg=self.config
        if not re.fullmatch('[0-9a-f]{40}',cfg['revision']):raise ValueError('Immutable model revision required')
        if cfg['generation']['do_sample'] or cfg['generation']['num_beams']!=1:raise ValueError('This stage requires greedy deterministic generation')
        self.device=torch.device(cfg['device'])
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('Configured CUDA unavailable; no silent fallback')
        self.dtype=getattr(torch,cfg['dtype'])
        cache=ROOT/cfg['cache_dir'];snapshot=cache/('models--'+cfg['model_id'].replace('/','--'))/'snapshots'/cfg['revision']
        if not snapshot.is_dir():raise FileNotFoundError('Pinned snapshot missing; run the documented download first')
        self.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(snapshot,dtype=self.dtype,device_map=str(self.device),attn_implementation=cfg['attention'],local_files_only=True,trust_remote_code=False)
        self.model.eval();self.model.requires_grad_(False)
        # Tokenizer advertises131072; this model config limits context to32768.
        self.context_limit=min(self.model.config.max_position_embeddings,self.tokenizer.model_max_length)
        self.input_limit=min(cfg['max_input_tokens'],self.context_limit-cfg['generation']['max_new_tokens'])
        if self.input_limit<=0 or cfg.get('truncation') is not False:raise ValueError('Require positive context budget and no truncation')
        self.runtime={'model_id':cfg['model_id'],'revision':cfg['revision'],'snapshot_path':str(snapshot),'device':str(self.model.device),'dtype':str(self.model.dtype),'attention':self.model.config._attn_implementation,'generation':cfg['generation'],'model_context_limit':self.model.config.max_position_embeddings,'tokenizer_model_max_length':self.tokenizer.model_max_length,'effective_context_limit':self.context_limit,'effective_input_limit':self.input_limit,'truncation':False,'parameter_count':self.model.num_parameters(),'chat_template_sha256':hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()}
    def moderate_input(self,request: str):return self.moderate('input',request)
    def moderate_output(self,request: str,response: str):return self.moderate('output',request,response)
    def moderate(self,task: str,request: str,response: str|None=None):
        import torch
        started=time.perf_counter();base={'task':task,'status':'error','parse_ok':False,'label':'unknown','categories':[],'refusal':None,'raw_output':None,'input_tokens':None,'generated_tokens':0,'truncated':False}
        try:
            if task not in {'input','output'} or not isinstance(request,str) or not request.strip():raise ValueError('invalid_task_or_request')
            if task=='output' and (not isinstance(response,str) or not response.strip()):raise ValueError('output_requires_nonempty_response')
            if task=='input' and response is not None:raise ValueError('input_task_must_not_have_response')
            messages=[{'role':'user','content':request}]
            if task=='output':messages.append({'role':'assistant','content':response})
            # Exact official workflow: no invented system text or generation suffix.
            text=self.tokenizer.apply_chat_template(messages,tokenize=False)
            inputs=self.tokenizer([text],return_tensors='pt',truncation=False)
            n=inputs['input_ids'].shape[-1];base.update(input_tokens=n,rendered_input_sha256=hashlib.sha256(text.encode()).hexdigest())
            if n>self.input_limit:
                base['error']={'code':'context_length_exceeded','input_tokens':n,'input_limit':self.input_limit,'max_new_tokens':self.config['generation']['max_new_tokens'],'context_limit':self.context_limit};return self._finish(base,started)
            inputs={k:v.to(self.device) for k,v in inputs.items()}
            self._sync();generation_start=time.perf_counter()
            base['generation_attempted']=True
            self.generation_calls=getattr(self,'generation_calls',0)+1
            with torch.inference_mode():generated=self.model.generate(**inputs,**self.config['generation'])
            self._sync();generation_seconds=time.perf_counter()-generation_start
            ids=generated[0,n:].tolist();raw=self.tokenizer.decode(ids,skip_special_tokens=True)
            base.update(parse_output(raw,task),generated_tokens=len(ids),generation_seconds=generation_seconds)
            eos=self.model.generation_config.eos_token_id
            eos={eos} if isinstance(eos,int) else set(eos or [])
            if not ids or ids[-1] not in eos:
                base.update(status='error',parse_ok=False,label='unknown',error={'code':'generation_incomplete_or_token_limit','max_new_tokens':self.config['generation']['max_new_tokens']})
            elif not base['parse_ok']:base['error']={'code':'parse_error','details':base['parse_errors']}
        except Exception as exc:
            base.update(status='error',parse_ok=False,label='unknown',error={'code':'moderation_exception','exception_type':type(exc).__name__,'message':str(exc)})
        return self._finish(base,started)
    def _sync(self):
        import torch
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    def _finish(self,result,started):
        result['policies']={name:apply_policy(result,labels) for name,labels in self.config['policies'].items()}
        result['detection_seconds']=time.perf_counter()-started
        return result
