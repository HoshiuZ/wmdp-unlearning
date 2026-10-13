"""Verify formal config, token cache, provenance and meta model without training."""
import sys,os,re,json,hashlib
from pathlib import Path
from hydra import initialize_config_dir,compose
from omegaconf import OmegaConf
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from accelerate import init_empty_weights
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'open-unlearning'
sys.path.insert(0,str(REPO/'src'));os.chdir(REPO)
from trainer import load_trainer_args
from data import get_data,get_collators
from data.pretraining import TokenBlockDataset
with initialize_config_dir(config_dir=str(REPO/'configs'),version_base=None):
 cfg=compose(config_name='train',overrides=['experiment=unlearn/wmdp/qwen3_4b_bio_formal'])
cfg.paths.work_dir=str(REPO)
args=load_trainer_args(cfg.trainer.args,None)
assert args.max_steps==80 and args.gradient_accumulation_steps==16 and args.per_device_train_batch_size==1
assert str(cfg.model.model_args.pretrained_model_name_or_path)==str(ROOT/'models/Qwen3-4B-Instruct-2507')
assert cfg.validation.selection_rule=='fixed_final_step_80'
assert not cfg.eval and not args.do_eval
config=AutoConfig.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507',local_files_only=True)
with init_empty_weights():model=AutoModelForCausalLM.from_config(config)
selected=[n for n,p in model.named_parameters() if any(re.fullmatch(r,n) for r in cfg.trainer.method_args.trainable_params_regex)]
assert selected==[f'model.layers.{i}.mlp.down_proj.weight' for i in (5,6,7)]
tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-4B-Instruct-2507',local_files_only=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
loaded=get_data(cfg.data,mode=cfg.mode,tokenizer=tokenizer,template_args=cfg.model.template_args)
collator=get_collators(cfg.collator,tokenizer=tokenizer);manifest=json.loads((ROOT/'logs/rmu_formal_blocks_manifest.json').read_text())
lengths={};examples={}
for info in manifest['files']:
 path=ROOT/info['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256']
 role,split=info['role'],info['split']
 dataset=getattr(loaded['train'],role) if split=='train' else TokenBlockDataset(
     {'path':'json','data_files':str(path),'split':'train'},cfg.model.template_args,tokenizer,max_length=256)
 assert len(dataset)==info['blocks'];lengths[f'{role}_{split}']=len(dataset)
 for index in range(len(dataset)):
  item=dataset[index];assert item['input_ids'].tolist()==dataset.data[index]['input_ids']
  assert 2<=len(item['input_ids'])<=256 and item['labels'][0]==-100
  assert (item['labels'][1:]==item['input_ids'][1:]).all()
 examples[role,split]=dataset.data[0]
# Independently tokenize representative source records and verify the selected slice.
for role in ['forget','retain']:
 for split in ['train','validation']:
  example=examples[role,split];source=ROOT/f'data/processed/bio_v1/{role}/{split}.jsonl';found=False
  for line in source.open():
   row=json.loads(line)
   if row['sample_id']==example['sample_id']:
    ids=tokenizer(row['text'],add_special_tokens=True)['input_ids']
    assert ids[example['token_start']:example['token_end']]==example['input_ids'];found=True;break
  assert found
batch=collator([loaded['train'][0]])
assert batch['forget']['index'].tolist()==[0]
resolved=OmegaConf.to_container(cfg,resolve=True)
(ROOT/'logs/rmu_formal_resolved_config.json').write_text(json.dumps(resolved,indent=2)+'\n')
report={'status':'prepared_not_started','training_started':False,'model_inspection':'meta tensors only',
        'base_checkpoint':str(ROOT/'models/Qwen3-4B-Instruct-2507'),'selected_parameters':selected,
        'dataset_lengths':lengths,'representative_source_token_slices_match':True,
        'all_cached_tokens_and_masks_verified':True,'sample_indices_available':True,
        'max_steps':80,'batch_per_device':1,'gradient_accumulation_steps':16,'effective_batch':16,
        'max_sequence_length':256,'learning_rate':args.learning_rate,'seed':args.seed,
        'validation_steps':[0,20,40,60,80],'checkpoint_selection':'fixed step80; no early stopping or test-based selection',
        'post_training_tests_used_for_tuning':False,'formal_cuda_run_verified':False,
        'short_run_cuda_evidence':'logs/rmu_smoke_run.json','cpu_validation_evidence':'logs/rmu_formal_monitor_cpu_checks.log',
        'token_block_manifest':'logs/rmu_formal_blocks_manifest.json'}
(ROOT/'logs/rmu_formal_readiness_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
