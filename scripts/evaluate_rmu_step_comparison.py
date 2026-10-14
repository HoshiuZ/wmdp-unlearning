"""Fixed full evaluations after continuous120, using one original evaluation model."""
import gc,hashlib,json,math,shutil,subprocess,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from rmu_adaptation_utils import ROOT,REPO,BASE,dump,score,load_delta,tensor_hash,base_samples
from rmu_step_common import RUN,STEPS,SELECTED,HISTORY,space
from run_rmu_formal import fingerprints,now

def checked_summary(data,reference):
 metrics={}
 for task,expected_n in [('wmdp_bio',1273),('mmlu',14042)]:
  original={(name,r['doc_id']):r for name,rs in reference[task].items() for r in rs};rows=data[task]
  assert len(rows)==expected_n and {(r['task'],r['doc_id']) for r in rows}==set(original)
  for r in rows:
   before=original[r['task'],r['doc_id']]
   assert (r['doc_hash'],r['prompt_hash'],r['target'])==(before['doc_hash'],before['prompt_hash'],before['doc']['answer'])
   assert len(r['scores'])==4 and all(math.isfinite(x) for x in r['scores'])
   assert r['prediction']==max(range(4),key=lambda k:r['scores'][k]) and r['acc']==int(r['prediction']==r['target'])
  correct=sum(r['acc'] for r in rows);b=sum(r['acc'] for r in original.values())
  metrics[task]={'n':len(rows),'correct':correct,'accuracy_percent':100*correct/len(rows),'baseline_accuracy_percent':100*b/len(rows),'change_percentage_points':100*(correct-b)/len(rows)}
 return metrics

def main():
 assert json.loads((RUN/'run.json').read_text())['status']=='completed','Do not evaluate an incomplete trajectory'
 out=RUN/'full_evaluations';out.mkdir(exist_ok=True);reference=base_samples();repro=json.loads((RUN/'reproduction40.json').read_text());need40=not repro['reuse_historical_full']
 started=time.perf_counter();evaluation={'started_at':now(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'steps':{},'new_full_evaluation_steps':[40,80,120] if need40 else [80,120]}
 dump(out/'evaluation_run.json',evaluation)
 model=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map='cuda');tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);model.requires_grad_(False);model.eval()
 expected=json.loads((REPO/'saves/unlearn/qwen3_4b_rmu_bio_smoke3_v1/base_weight_fingerprints.json').read_text());assert fingerprints(model)==expected
 original={n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in SELECTED}
 requesthash=hashlib.sha256(json.dumps({task:{name:[r['arguments'] for r in rs] for name,rs in subjects.items()} for task,subjects in reference.items()},sort_keys=True).encode()).hexdigest()
 for step in STEPS:
  dest=out/f'checkpoint-{step}';assert not dest.exists(),'Refuse to overwrite prior evaluation'
  space(200*2**20);checkpoint=RUN/f'checkpoint-{step}';begin=time.perf_counter()
  # Every load computes XOR against the SAME immutable original dictionary.
  load_delta(model,checkpoint,original)
  assert all(tensor_hash(dict(model.named_parameters())[r['name']])==r['trained_tensor_sha256'] for r in json.loads((checkpoint/'checkpoint.json').read_text())['parameters'])
  reused=step==40 and not need40
  config={'checkpoint':str(checkpoint.relative_to(ROOT)),'num_fewshot':0,'apply_chat_template':False,'system_instruction':None,'batch_size':1,'dtype':'bfloat16','attention':'sdpa','task_sizes':{'wmdp_bio':1273,'mmlu':14042},'exact_archived_baseline_requests_sha256':requesthash,'scoring':'argmax of archived four-option continuation loglikelihood requests; no generation','exploratory':True,'reused_historical_predictions':reused,'delta_always_against_fixed_original':True}
  dump(dest/'evaluation_config.json',config)
  if reused:
   source=ROOT/repro['historical_full_source'];shutil.copy2(source/'samples.json',dest/'samples.json')
   data=json.loads((dest/'samples.json').read_text());metrics=checked_summary(data,reference)
   dump(dest/'summary.json',{'full':True,'exploratory':True,'seconds':0,'reused':True,'source':str(source.relative_to(ROOT)),'metrics':metrics,'evidence':'Step40 parameter hashes identical; pinned tokenizer; exact archived questions/prompts/targets and request protocol verified'})
  else:
   score(model,tok,dest,full=True);data=json.loads((dest/'samples.json').read_text());metrics=checked_summary(data,reference)
   recorded=json.loads((dest/'summary.json').read_text());assert metrics==recorded['metrics']
  dump(dest/'result.json',{'step':step,'metrics':metrics,'reused':reused,'wall_seconds_including_reload_and_validation':time.perf_counter()-begin})
  evaluation['steps'][str(step)]=json.loads((dest/'result.json').read_text());dump(out/'evaluation_run.json',evaluation)
  print('FULL_COMPLETE',step,metrics,flush=True)
 evaluation.update(completed_at=now(),total_wall_seconds_including_loading_and_checks=time.perf_counter()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved());dump(out/'evaluation_run.json',evaluation)
 print('ALL_FIXED_EVALUATIONS_COMPLETE',flush=True)
if __name__=='__main__':main()
