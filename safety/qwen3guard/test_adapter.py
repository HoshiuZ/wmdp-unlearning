"""Failure-path checks: no classifier weights or RMU models are loaded."""
import json,unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from .moderator import Qwen3Guard,parse_output,apply_policy

class AdapterChecks(unittest.TestCase):
    def test_label_policy_separation(self):
        audit=parse_output('Safety: Controversial\nCategories: Violent','input')
        self.assertTrue(audit['parse_ok'])
        self.assertFalse(apply_policy(audit,['Unsafe'])['blocked'])
        self.assertTrue(apply_policy(audit,['Unsafe','Controversial'])['blocked'])
    def test_official_response_fields(self):
        audit=parse_output('Safety: Safe\nCategories: None\nRefusal: Yes','output')
        self.assertTrue(audit['parse_ok']);self.assertEqual(audit['refusal'],'Yes')
        risky=parse_output('Safety: Unsafe\nCategories: Violent\nRefusal: Yes','output')
        self.assertTrue(apply_policy(risky,['Unsafe'])['blocked'])
    def test_invalid_results_never_allow(self):
        cases=['','Safety: Safe','Safety: Maybe\nCategories: None','Safety: Safe\nSafety: Unsafe\nCategories: None','Safety: Safe\nCategories: Violent','Safety: Unsafe\nCategories: Unknown','Safety: Safe\nCategories: None, Violent','Some prose\nSafety: Safe\nCategories: None']
        for raw in cases:
            with self.subTest(raw=raw):
                audit=parse_output(raw,'input');self.assertFalse(audit['parse_ok']);self.assertEqual(audit['label'],'unknown')
                self.assertEqual(apply_policy(audit,['Unsafe'])['action'],'hold_for_review')
        self.assertFalse(parse_output('Safety: Safe\nCategories: None','output')['parse_ok'])
        self.assertFalse(parse_output('Safety: Unsafe\nCategories: Jailbreak\nRefusal: No','output')['parse_ok'])
    @staticmethod
    def fake_guard(length=2,raises=False):
        class Tokenizer:
            def apply_chat_template(self,*args,**kwargs):return 'official-template-placeholder'
            def __call__(self,*args,**kwargs):return {'input_ids':torch.ones((1,length),dtype=torch.long)}
            def decode(self,*args,**kwargs):return 'Safety: Safe\nCategories: None'
        class Model:
            calls=0;generation_config=SimpleNamespace(eos_token_id=[0])
            def generate(self,**kwargs):
                self.calls+=1
                if raises:raise RuntimeError('injected_generation_failure')
                return torch.ones((1,length+1),dtype=torch.long)
        guard=Qwen3Guard.__new__(Qwen3Guard);guard.config=json.loads(Path(__file__).with_name('config.json').read_text());guard.device=torch.device('cpu');guard.context_limit=20;guard.input_limit=10;guard.tokenizer=Tokenizer();guard.model=Model();return guard
    def test_overflow_skips_generation(self):
        guard=self.fake_guard(length=11);result=guard.moderate_input('brief query')
        self.assertEqual(result['error']['code'],'context_length_exceeded');self.assertEqual(guard.model.calls,0)
        self.assertEqual(result['policies']['main']['action'],'hold_for_review')
    def test_generation_exception_and_incomplete_output(self):
        for raises in [False,True]:
            guard=self.fake_guard(raises=raises);result=guard.moderate_input('brief query')
            self.assertEqual(result['label'],'unknown');self.assertEqual(result['policies']['strict']['action'],'hold_for_review')
            self.assertEqual(guard.model.calls,1)
    def test_missing_response(self):
        guard=self.fake_guard();result=guard.moderate('output','query')
        self.assertEqual(result['status'],'error');self.assertEqual(guard.model.calls,0)

if __name__=='__main__':unittest.main()
