"""Verify accumulated gradients, cache ingestion and held-out monitor on a tiny CPU model."""
import json,sys,tempfile
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM,TrainingArguments
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'open-unlearning/src'));sys.path.insert(0,str(ROOT/'scripts'))
from trainer.unlearn.rmu import RMU
from data.pretraining import TokenBlockDataset
from data.collators import DataCollatorForSupervisedDataset
from rmu_validation import HeldOutValidation
class TokenizerStub:
    pad_token_id=31
    def __len__(self):return 32
with tempfile.TemporaryDirectory() as directory:
    path=Path(directory);source=path/'blocks.jsonl'
    source.write_text(''.join(json.dumps({'input_ids':ids,'sample_id':str(i)})+'\n' for i,ids in enumerate([[1,2,3,4],[5,6,7]])))
    tokenizer=TokenizerStub()
    dataset=TokenBlockDataset({'path':'json','data_files':str(source),'split':'train'},None,tokenizer,max_length=256)
    assert dataset[0]['input_ids'].tolist()==[1,2,3,4] and dataset[0]['labels'].tolist()==[-100,2,3,4]
    collator=DataCollatorForSupervisedDataset(tokenizer,index='index')
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=8,
                      num_attention_heads=2,num_key_value_heads=2,head_dim=8)
    model=Qwen3ForCausalLM(config)
    args=TrainingArguments(output_dir=str(path/'run'),use_cpu=True,report_to=[],optim='adamw_torch')
    trainer=RMU(model=model,args=args,retain_loss_type='EMBED_DIFF',steering_coeff=2,
                module_regex=r'model\.layers\.7',trainable_params_regex=[r'model\.layers\.(5|6|7)\.mlp\.down_proj\.weight'])
    trainer.create_optimizer()
    pair=collator([{'forget':dataset[0],'retain':dataset[1]}])
    trainer.compute_loss(model,pair).backward()
    reference={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True);trainer.current_gradient_accumulation_steps=16
    for _ in range(16):trainer.training_step(model,pair,num_items_in_batch=torch.tensor(100))
    for n,p in model.named_parameters():
        if p.requires_grad:assert torch.allclose(p.grad,reference[n],rtol=2e-5,atol=2e-6),n
    model.zero_grad(set_to_none=True);trainer.step_loss_records=[]
    before={n:p.clone() for n,p in model.named_parameters()}
    monitor=HeldOutValidation(trainer,dataset,dataset,collator,path/'monitor',every_steps=20)
    monitor.evaluate(0,0)
    assert model.training and not trainer.step_loss_records
    assert all(p.grad is None and torch.equal(p,before[n]) for n,p in model.named_parameters())
    result=monitor.results[0];assert result['blocks_per_role']==2 and result['nll_tokens']==5
    assert result['retain_loss']==0 and result['retain_nll']>0
    assert pair['forget']['index'].tolist()==[0] and pair['retain']['index'].tolist()==[1]
print('PASS: cached tokens preserved; sample indices; 16-step accumulation matches one-batch gradient; validation no gradients/parameter changes; weighted NLL and mode restored')
