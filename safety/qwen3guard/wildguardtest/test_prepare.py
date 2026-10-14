import unittest
from safety.qwen3guard.wildguardtest.prepare import build


class PreparationTests(unittest.TestCase):
    def test_conflicts_missing_and_independent_output(self):
        def row(p, ph, rh, response='synthetic response'):
            return dict(prompt=p, prompt_harm_label=ph, response_harm_label=rh,
                        response=response, response_refusal_label=None, adversarial=False)
        rows = [row('conflict', 'harmful', 'unharmful'), row('conflict', 'unharmful', 'harmful'),
                row('known', 'harmful', None), row('known', None, 'unharmful'), row('missing', None, None)]
        inputs, outputs, conflicts, summary = build(rows)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual([(x['gold'], x['rows']) for x in inputs], [('harmful', [2, 3])])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(summary['input_without_known_label'], 1)

    def test_unexpected_label_stops(self):
        with self.assertRaises(ValueError):
            build([dict(prompt='synthetic', prompt_harm_label='uncertain', response_harm_label=None)])


if __name__ == '__main__':
    unittest.main()
