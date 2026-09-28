"""
test_attack_taxonomy.py
=======================
Guards attack_taxonomy.py, the single source of ATT&CK labels.

The failures it protects against were real: tactics were stored in a column
called `attack_technique`, three dictionaries gave the same API call different
tactics, and a detonation's single tactic was stamped on every event in it
(making AssumeRole "credential-access", which ATT&CK does not allow for
T1078.004).
"""

import re
import unittest

import attack_taxonomy as at

TECHNIQUE_ID = re.compile(r"^T\d{4}(\.\d{3})?$")


class TestTaxonomyDefinitions(unittest.TestCase):
    def test_tactic_ids_are_attack_tactic_ids(self):
        for slug, tactic in at.TACTICS.items():
            self.assertRegex(tactic["id"], r"^TA\d{4}$", slug)

    def test_technique_ids_are_technique_ids_not_tactics(self):
        for tid in at.TECHNIQUES:
            self.assertRegex(tid, TECHNIQUE_ID)
            self.assertNotIn(tid, at.TACTICS)

    def test_every_technique_lists_known_tactics_and_a_basis(self):
        for tid, tech in at.TECHNIQUES.items():
            self.assertTrue(tech["tactics"], tid)
            self.assertLessEqual(tech["tactics"], set(at.TACTICS), tid)
            self.assertTrue(tech["basis"].strip(), f"{tid} has no written mapping rationale")

    def test_unmapped_events_carry_a_reason(self):
        for event, reason in at.UNMAPPED_REASONS.items():
            self.assertTrue(reason.strip(), event)


class TestValidate(unittest.TestCase):
    def test_consistent_pair_passes(self):
        at.validate("privilege-escalation", "T1078.004")
        at.validate("credential-access", "T1555.006")

    def test_benign_row_passes(self):
        at.validate(None, None)

    def test_unmapped_is_allowed_with_any_tactic(self):
        at.validate("persistence", at.UNMAPPED)

    def test_tactic_outside_the_techniques_tactics_is_rejected(self):
        # The exact inconsistency the old per-detonation labeling produced.
        with self.assertRaises(ValueError):
            at.validate("credential-access", "T1078.004")

    def test_unknown_technique_is_rejected(self):
        with self.assertRaises(ValueError):
            at.validate("persistence", "T9999")

    def test_a_tactic_is_not_accepted_as_a_technique(self):
        with self.assertRaises(ValueError):
            at.validate("persistence", "privilege-escalation")

    def test_unknown_tactic_is_rejected(self):
        with self.assertRaises(ValueError):
            at.validate("Persistence", "T1136.003")  # slug, not display name

    def test_technique_without_tactic_is_rejected(self):
        with self.assertRaises(ValueError):
            at.validate(None, "T1136.003")


class TestLegacyInvictusDictionary(unittest.TestCase):
    def test_every_entry_is_a_valid_pair(self):
        for event, (tactic, tech) in at.INVICTUS_EVENT_LABELS.items():
            at.validate(tactic, tech)

    def test_each_event_has_exactly_one_label(self):
        # A dict cannot hold duplicates, but the old notebook dictionary and the
        # generator disagreed; this pins one answer per event in one place.
        self.assertEqual(len(at.INVICTUS_EVENT_LABELS), len(set(at.INVICTUS_EVENT_LABELS)))


if __name__ == "__main__":
    unittest.main()
