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



class TestCausalBaseRatePrior(unittest.TestCase):
    """The tests above use a fixed default (0.1). FeatureEngineer uses
    default=AdaptiveRiskPrior.BASE_RATE: every key shrinks toward the fitted
    TRAINING attack rate. That deliberately lets one key's labels move another
    key's score -- but only through that single global rate, still only from
    labels seen earlier, and never once frozen."""

    W = 15

    def prior(self):
        return AdaptiveRiskPrior(priors={}, default=AdaptiveRiskPrior.BASE_RATE,
                                 prior_weight=self.W, frozen=False)

    def test_unseen_key_scores_exactly_the_base_rate(self):
        p = self.prior()
        for label in ("1", "0", "0", "0"):
            p.update("keyA", label)
        self.assertAlmostEqual(p.base_rate(), 0.25)
        self.assertAlmostEqual(p.score("keyB"), 0.25)

    def test_other_keys_influence_is_only_through_the_global_rate(self):
        """Two histories with the same overall rate but different per-key
        detail give an unseen key the same score: no per-key information leaks."""
        a, b = self.prior(), self.prior()
        for label in ("1", "0"):
            a.update("keyA", label)
        a.update("keyC", "0"); a.update("keyC", "1")
        for label in ("1", "1"):
            b.update("keyX", label)
        b.update("keyY", "0"); b.update("keyY", "0")
        self.assertAlmostEqual(a.score("keyB"), b.score("keyB"))

    def test_seen_key_blends_its_own_rate_with_the_base_rate(self):
        p = self.prior()
        for _ in range(5):
            p.update("keyA", "1")
        for _ in range(15):
            p.update("keyB", "0")
        base = 5 / 20
        self.assertAlmostEqual(p.score("keyA"), (self.W * base + 5) / (self.W + 5))

    def test_score_before_update_excludes_own_label(self):
        correct, leaky = self.prior(), self.prior()
        for p in (correct, leaky):
            p.update("keyZ", "0")
        feature = correct.score("AssumeRole")
        correct.update("AssumeRole", "1")
        leaky.update("AssumeRole", "1")
        self.assertLess(feature, leaky.score("AssumeRole"))

    def test_frozen_base_rate_does_not_move(self):
        p = self.prior()
        p.update("keyA", "1"); p.update("keyA", "0")
        p.frozen = True
        before = (p.base_rate(), p.score("keyA"), p.score("unseen"))
        for _ in range(100):
            p.update("keyB", "1")
        self.assertEqual(before, (p.base_rate(), p.score("keyA"), p.score("unseen")))

    def test_feature_engineer_uses_the_base_rate_setting(self):
        from feature_engine9 import FeatureEngineer
        engine = FeatureEngineer()
        self.assertEqual(engine.action_risk_prior.default, AdaptiveRiskPrior.BASE_RATE)
        self.assertEqual(engine.principal_risk_prior.default, AdaptiveRiskPrior.BASE_RATE)


if __name__ == "__main__":
    unittest.main()
