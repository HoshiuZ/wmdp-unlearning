import unittest
from safety.qwen3guard.wildguardtest.evaluate import metrics,wilson,distribution
from safety.qwen3guard.moderator import apply_policy

class MetricTests(unittest.TestCase):
    def test_confusion_and_class_f1(self):
        result=metrics([(True,True),(True,False),(False,True),(False,False)])
        self.assertEqual([result[x] for x in ('TP','FP','TN','FN')],[1,1,1,1])
        for key in ('precision','recall','F1','macro_F1','FPR','FNR'):
            self.assertEqual(result[key],.5)
        lo,hi=result['recall_CI95'];self.assertLess(lo,.5);self.assertGreater(hi,.5)

    def test_unknown_is_blocked_for_both_policies(self):
        unknown={'status':'error','parse_ok':False,'label':'unknown'}
        for labels in (['Unsafe'],['Unsafe','Controversial']):
            decision=apply_policy(unknown,labels)
            self.assertTrue(decision['blocked'])
            self.assertEqual(decision['action'],'hold_for_review')
        # Fail-closed recovers a harmful failure and also blocks a benign failure.
        result=metrics([(True,True),(False,True)])
        self.assertEqual(result['recall'],1)
        self.assertEqual(result['FPR'],1)

    def test_empty_and_boundary_intervals(self):
        self.assertIsNone(metrics([])['recall'])
        self.assertIsNone(wilson(0,0))
        self.assertAlmostEqual(wilson(0,10)[0],0)
        self.assertAlmostEqual(wilson(10,10)[1],1)
        self.assertEqual(distribution([1,2,3])['P50'],2)

if __name__=='__main__':unittest.main()
