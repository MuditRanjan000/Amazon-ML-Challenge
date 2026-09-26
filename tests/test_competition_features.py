import unittest

import pandas as pd

from entity_resolution.features import CompetitionFeatureConfig, add_competition_features
from entity_resolution.models.decisions import OneOwnerThresholdDecisionLayer


class CompetitionFeatureTests(unittest.TestCase):
    def test_v2_features_join_by_candidate_without_filtering_nonowners(self):
        base = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2", "S1-1"],
                "candidate_entity_id": ["S2-1", "S2-1", "S3-1"],
                "pair_local_evidence": [0.8, 0.7, 0.9],
            }
        )
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2", "S1-1"],
                "candidate_entity_id": ["S2-1", "S2-1", "S3-1"],
                "score": [0.9, 0.6, 0.8],
            }
        )
        stats = pd.DataFrame(
            {
                "candidate_entity_id": ["S2-1", "S3-1"],
                "best_score": [0.9, 0.8],
                "second_score": [0.6, float("nan")],
                "n_s1": [2, 1],
                "best_s1": ["S1-1", "S1-1"],
            }
        )
        actual, columns = add_competition_features(base, pairs, stats)
        self.assertEqual(len(actual), 3)
        self.assertIn("owner_gap", columns)
        self.assertEqual(actual.loc[1, "is_best_s1"], 0)
        self.assertAlmostEqual(actual.loc[1, "margin_to_best"], 0.3)
        self.assertAlmostEqual(actual.loc[2, "second_score"], -1.0)
        self.assertAlmostEqual(actual.loc[2, "owner_gap"], 1.8)

    def test_v2_features_reject_incomplete_stats_by_default(self):
        base = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"], "x": [1.0]})
        pairs = pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"], "score": [0.5]})
        stats = pd.DataFrame(
            columns=["candidate_entity_id", "best_score", "second_score", "n_s1", "best_s1"]
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            add_competition_features(base, pairs, stats, CompetitionFeatureConfig())

    def test_one_owner_uses_probability_tie_breaking_and_keeps_multiple_per_source(self):
        scores = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2", "S1-1", "S1-3"],
                "candidate_entity_id": ["S2-1", "S2-1", "S3-1", "S3-1"],
                "match_probability": [0.8, 0.8, 0.9, 0.85],
            }
        )
        decisions = OneOwnerThresholdDecisionLayer(0.75, "match_probability").decide(
            scores, ["S1-1", "S1-2", "S1-3", "S1-4"]
        )
        got = decisions.set_index("source1_entity_id")["matched_entity_ids"].to_dict()
        self.assertEqual(got["S1-1"], "S2-1,S3-1")
        self.assertEqual(got["S1-2"], "")
        self.assertEqual(got["S1-3"], "")
        self.assertEqual(got["S1-4"], "")


if __name__ == "__main__":
    unittest.main()
