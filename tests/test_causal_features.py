"""
Causal-feature / temporal-leakage tests (review point 2).

The stateful label-derived feature (source_historical_risk, via
AdaptiveRiskPrior) is target encoding: score(key) is a function of labels seen
SO FAR. That is legitimate only if it is strictly causal -- feature_t depends on
events <= t-1, never on future events or the row's own label -- and read-only at
evaluation. These tests assert both properties, so a regression that reintroduces
leakage fails CI rather than silently inflating results.
"""

import unittest

from feature_engine9 import AdaptiveRiskPrior


class TestCausalRiskPrior(unittest.TestCase):
    def test_score_excludes_own_and_future_updates(self):
        """score(key) at 'time t' reflects only update()s applied before it --
        an update applied AFTER a score was taken cannot have influenced it, and
        an update to one key never leaks into another key's score."""
        p = AdaptiveRiskPrior(priors={}, default=0.1, prior_weight=15, frozen=False)

        # scoring keyB is unaffected by any number of updates to keyA (no
        # cross-key leakage, and keyB has seen no labels of its own yet)
        b_before = p.score("keyB")
        for _ in range(50):
            p.update("keyA", "1")
        b_after = p.score("keyB")
        self.assertEqual(b_before, b_after,
                         "another key's updates leaked into this key's score")

        # a key's score only rises AFTER its own attack labels are folded in;
        # the value taken before the update did not see it
        a0 = p.score("keyA_fresh")
        p.update("keyA_fresh", "1")
        a1 = p.score("keyA_fresh")
        self.assertGreater(a1, a0,
                           "target encoding not updating (should rise after an attack label)")

    def test_causal_ordering_avoids_self_leakage(self):
        """The pipeline computes the feature (score) BEFORE folding the row's own
        label (update). If the order were reversed, a row's own label would leak
        into its own feature. This asserts the correct order produces a strictly
        lower score than the leaky order."""
        key = "AssumeRole"
        correct = AdaptiveRiskPrior(priors={}, frozen=False)
        leaky = AdaptiveRiskPrior(priors={}, frozen=False)

        # correct order: score first (feature), then observe the label
        correct_feature = correct.score(key)
        correct.update(key, "1")

        # leaky order: observe the label first, then score
        leaky.update(key, "1")
        leaky_feature = leaky.score(key)

        self.assertLess(correct_feature, leaky_feature,
                        "score-before-update must exclude the row's own label")

    def test_frozen_prior_is_readonly_on_eval(self):
        """A frozen prior (any non-training input) ignores every update, so
        evaluation labels can never enter the features -- the same discipline as
        a fit-then-transform scaler. Fit on 'training' first, then freeze and
        confirm 'evaluation' labels move nothing."""
        prior = AdaptiveRiskPrior(priors={}, frozen=False)   # fit on training
        prior.update("GetSecretValue", "1")
        prior.update("GetSecretValue", "0")
        prior.frozen = True                                  # now an eval run
        before = prior.score("GetSecretValue")
        for _ in range(100):
            prior.update("GetSecretValue", "1")              # must all be no-ops
        after = prior.score("GetSecretValue")
        self.assertEqual(before, after,
                         "frozen prior changed under update -- evaluation label leakage")


if __name__ == "__main__":
    unittest.main()
