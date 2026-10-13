"""CPU checks for RMU freeze mask, gradient flow, masking, and failed-hook cleanup."""
import sys
from pathlib import Path
import tempfile
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "open-unlearning/src"))
from trainer.unlearn.rmu import RMU

config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                    num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2, head_dim=8)
model = Qwen3ForCausalLM(config)
with tempfile.TemporaryDirectory() as directory:
    args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[], optim="adamw_torch")
    trainer = RMU(model=model, args=args, retain_loss_type="EMBED_DIFF", steering_coeff=2,
                  module_regex=r"model\.layers\.7",
                  trainable_params_regex=[r"model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"])
    trainer.create_optimizer()
    names = [n for n,p in model.named_parameters() if p.requires_grad]
    assert names == [f"model.layers.{i}.mlp.down_proj.weight" for i in (5,6,7)], names
    assert not any(p.requires_grad for p in trainer.ref_model.parameters())
    assert {id(p) for g in trainer.optimizer.param_groups for p in g["params"]} == {id(p) for p in model.parameters() if p.requires_grad}
    ids = torch.randint(0,32,(1,8)); labels=ids.clone();labels[:,0]=-100
    batch={"input_ids":ids,"attention_mask":torch.ones_like(ids),"labels":labels}
    loss=trainer.compute_loss(model,{"forget":batch,"retain":batch})
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    a=torch.tensor([[[1.],[1.],[1.]],[[3.],[3.],[3.]]],requires_grad=True)
    mask=torch.tensor([[True,False,False],[True,True,True]])
    value=trainer.compute_activation_loss(a,torch.zeros_like(a),mask)
    assert torch.allclose(value,torch.tensor(5.)),value
    value.backward();assert a.grad[0,1:].abs().sum()==0
    try: trainer.compute_activation_loss(a,torch.zeros_like(a),torch.zeros_like(mask))
    except ValueError: pass
    else: raise AssertionError("All-masked example accepted")
    class Broken(torch.nn.Module):
        def __init__(self): super().__init__();self.part=torch.nn.Identity()
        def forward(self,x): self.part(x);raise RuntimeError("intentional test failure")
    broken=Broken()
    try: trainer.forward_with_cache(broken,{"x":torch.ones(1)},broken.part)
    except RuntimeError: pass
    assert not broken.part._forward_hooks
print("PASS: selected optimizer/gradient mask; frozen reference; finite backward; batched masked loss; empty-mask guard; hook cleanup")
