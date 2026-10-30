"""Small diagnostic checks; no training or original evidence writes."""
import unittest

import numpy as np
from neural_null import MODEL_SEEDS, PERMUTATION_SEEDS, ensemble_score, interpret, shuffled_role


class MatchedNullChecks(unittest.TestCase):
    def test_fixed_seed_roster_excludes_historical_permutation(self):
        self.assertEqual(PERMUTATION_SEEDS, tuple(range(1, 201)))
        self.assertEqual(MODEL_SEEDS, (0, 1, 2, 3, 4))

    def test_role_shuffle_preserves_inputs_and_label_multiset(self):
        items = [{'video_id': str(i), 'mos': float(i)} for i in range(12)]
        shuffled, mapping = shuffled_role(items, 1)
        self.assertEqual([r['mos'] for r in items], list(map(float, range(12))))
        self.assertEqual(sorted(r['mos'] for r in shuffled), list(map(float, range(12))))
        self.assertEqual([r['video_id'] for r in shuffled], [r['video_id'] for r in items])
        self.assertEqual([r['mos'] for r in shuffled], [float(m['label_source_id']) for m in mapping])
        self.assertEqual(shuffled_role(items, 1), (shuffled, mapping))

    def test_interpretation_inside_outside_boundary_and_partial(self):
        scores = np.linspace(-.2, .2, 200)
        interval, verdict = interpret(scores, -.1563)
        self.assertEqual(verdict, 'consistent with chance under the control’s own procedure')
        self.assertEqual(interpret(scores, interval[0])[1], verdict)
        self.assertEqual(interpret(scores, interval[1])[1], verdict)
        self.assertEqual(interpret(scores, -.3)[1], 'not explained by chance; investigate')
        with self.assertRaises(ValueError):
            interpret(scores[:-1], -.1563)

    def test_averages_predictions_before_ranking(self):
        predictions = [[0, 1, 2]] * 4 + [[100, 0, -100]]
        values, score = ensemble_score(predictions, [0, 1, 2])
        np.testing.assert_allclose(values, [20, .8, -18.4])
        self.assertAlmostEqual(score, -1)
        with self.assertRaises(ValueError):
            ensemble_score(predictions[:4], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
