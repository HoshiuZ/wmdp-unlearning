from pathlib import Path
import sys,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'open-unlearning';sys.path.insert(0,str(REPO/'src'))
import os
os.chdir(REPO)
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from accelerate import init_empty_weights
from trainer import load_trainer_args
from data import get_data,get_collators
with initialize_config_dir(config_dir=str(REPO/'configs'),version_base=None):
 cfg=compose(config_name='train',overrides=['experiment=unlearn/wmdp/qwen3_4b_bio_smoke'])
cfg.paths.work_dir=str(REPO)
args=load_trainer_args(cfg.trainer.args,None)
assert args.max_steps==3 and not args.do_eval and args.remove_unused_columns is False
config=AutoConfig.from_pretrained(str(ROOT/'models/Qwen3-4B-Instruct-2507'),local_files_only=True)
with init_empty_weights(): model=AutoModelForCausalLM.from_config(config)
patterns=cfg.trainer.method_args.trainable_params_regex
selected=[{'name':n,'shape':list(p.shape),'parameters':p.numel()} for n,p in model.named_parameters() if any(re.fullmatch(r,n) for r in patterns)]
modules=[n for n,_ in model.named_modules() if re.fullmatch(cfg.trainer.method_args.module_regex,n)]
assert len(selected)==3 and modules==['model.layers.7']
assert all(p.device.type=='meta' for p in model.parameters())
tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-4B-Instruct-2507'),local_files_only=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
loaded=get_data(cfg.data,mode=cfg.mode,tokenizer=tokenizer,template_args=cfg.model.template_args)
dataset=loaded['train'];assert len(dataset)==8
for role in ['forget','retain']:
 d=getattr(dataset,role)
 source=ROOT/f'data/processed/bio_v1/smoke_input/{role}.jsonl'
 first=json.loads(source.read_text().splitlines()[0]);assert d.data[0]['text']==first['text']
 for i in range(len(d)):
  item=d[i];assert 2<=len(item['input_ids'])<=256;assert (item['labels']!=-100).any()
collator=get_collators(cfg.collator,tokenizer=tokenizer)
batch=collator([dataset[0]])
shapes={role:{key:list(value.shape) for key,value in entry.items()} for role,entry in batch.items()}
resolved=OmegaConf.to_container(cfg,resolve=True)
(ROOT/'logs/rmu_smoke_resolved_config.json').write_text(json.dumps(resolved,indent=2)+'\n')
r={'module_matches':modules,'selected_parameters':selected,'trainable_parameter_count':sum(p['parameters'] for p in selected),'total_model_parameters':sum(p.numel() for p in model.parameters()),'model_inspection':'meta tensors only; no 4B weights loaded','reference_model':'RMU makes deepcopy on accelerator device; frozen after preparation','dataset_lengths':{'forget':len(dataset.forget),'retain':len(dataset.retain)},'collated_tensor_shapes':shapes,'json_text_matches_source':True,'all_examples_valid_labels':True,'max_length':256,'model_dtype_for_actual_run':'bfloat16','short_run_max_steps':3,'short_run_started':False,'cuda_training_verified':False,'subset_manifest':'logs/rmu_smoke_subset_manifest.json','cpu_checks':'logs/rmu_cpu_checks.log','loss_fix_note':'Batch>1 normalization previously broadcast per-example sums with [b,1] denominator; corrected to [b]. Batch1 formula preserved.','freeze_fix_note':'Keep only selected parameters requires_grad after optimizer construction; freeze reference parameters explicitly.'}
(ROOT/'logs/rmu_readiness_report.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
