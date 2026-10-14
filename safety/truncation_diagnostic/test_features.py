import unittest
from .summarize import features
class FeatureTests(unittest.TestCase):
 def test_fences_and_control_markers(self):
  self.assertFalse(features('```text\nExample\n```')['unclosed_code_fence'])
  self.assertTrue(features('```text\nExample')['unclosed_code_fence'])
  self.assertTrue(features('<think>example</think>')['format_control_or_think'])
 def test_repetition_and_tail_are_only_signals(self):
  r=features(('This line is intentionally repeated.\n')*20)
  self.assertTrue(r['high_fourgram_repetition']);self.assertTrue(r['repeated_long_lines'])
  self.assertTrue(features('The next step is to')['ends_with_connective'])
  self.assertFalse(features('A complete sentence.')['nonterminal_tail'])
if __name__=='__main__':unittest.main()
