import unittest
from .policy import outcome
from .common import digest
from safety.qwen3guard.moderator import apply_policy

def audit(label=None):
 a={'status':'ok' if label else 'error','parse_ok':label is not None,'label':label or 'unknown'}
 a['policies']={k:apply_policy(a,v) for k,v in {'main':['Unsafe'],'strict':['Unsafe','Controversial']}.items()};return a
class FlowTests(unittest.TestCase):
 def test_three_paths(self):
  good={'status':'ok'}
  self.assertEqual(outcome(audit('Unsafe'),None,None,'main'),'input_block')
  self.assertEqual(outcome(audit('Safe'),good,audit('Unsafe'),'main'),'output_block')
  self.assertEqual(outcome(audit('Safe'),good,audit('Safe'),'main'),'allow')
 def test_unknown_and_failed_generation_are_exceptions(self):
  self.assertEqual(outcome(audit(),None,None,'main'),'exception_input')
  self.assertEqual(outcome(audit('Safe'),{'status':'error'},None,'main'),'exception_generation')
  self.assertEqual(outcome(audit('Safe'),{'status':'ok'},audit(),'main'),'exception_output')
  self.assertEqual(outcome(audit('Unsafe'),{'status':'error'},None,'main'),'input_block')
 def test_policies_share_levels_not_unknown(self):
  self.assertEqual(outcome(audit('Controversial'),{'status':'ok'},audit('Safe'),'main'),'allow')
  self.assertEqual(outcome(audit('Controversial'),{'status':'ok'},audit('Safe'),'strict'),'input_block')
 def test_content_namespace(self):
  self.assertNotEqual(digest(['input','a']),digest(['output','a']))
  self.assertNotEqual(digest(['ab','c']),digest(['a','bc']))
if __name__=='__main__':unittest.main()
