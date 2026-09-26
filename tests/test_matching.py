import tempfile
import unittest
import json
import importlib.util
import pickle
import gzip
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.feature_extraction.text import TfidfVectorizer

from src.entity_resolution.matching.contracts import validate_candidate_pairs
from src.entity_resolution.matching.decision import ThresholdDecisionLayer
from src.entity_resolution.matching.features import PairwiseFeatureExtractor
from src.entity_resolution.matching.normalization import (
    _python_jaro_winkler_similarity_normalized,
    _python_levenshtein_distance_normalized,
    jaro_winkler_similarity_normalized,
    levenshtein_distance_normalized,
    jaro_winkler_similarity,
    normalize_for_matching,
    token_set,
)
from src.entity_resolution.matching.records import DataFrameRecordAdapter, SQLiteRecordStore
from src.entity_resolution.matching.scoring import DeterministicScorer
from src.entity_resolution.models.experiments import BoundedExperimentConfig, run_bounded_experiment
from src.entity_resolution.features import PairwiseFeatureExtractor as OwnedPairwiseFeatureExtractor
from src.entity_resolution.models import DeterministicScorer as OwnedDeterministicScorer
from src.entity_resolution.models.logistic import fit_logistic_regression, predict_match_probabilities
from src.entity_resolution.models.experiments import (
    select_ground_truth,
    select_stable_stratified_train_ids,
    validate_frozen_split_manifest,
    verify_repository_evaluator_known_answer,
)
from src.entity_resolution.models.streaming import (
    CandidatePairSpool,
    StreamingConfig,
    assemble_score_output,
    build_threshold_decisions,
    compute_disk_backed_error_counts,
    evaluate_with_shared_evaluator,
    score_logistic_stream,
)

TEMP_CANDIDATE_SPEC = importlib.util.spec_from_file_location(
    "temp_exact_candidates", Path(__file__).resolve().parents[1] / "execution" / "generate_temp_exact_candidates.py"
)
assert TEMP_CANDIDATE_SPEC and TEMP_CANDIDATE_SPEC.loader
temp_exact_candidates = importlib.util.module_from_spec(TEMP_CANDIDATE_SPEC)
TEMP_CANDIDATE_SPEC.loader.exec_module(temp_exact_candidates)


def source_frames():
    source1 = pd.DataFrame(
        [
            {
                "entity_id": "S1-1",
                "business_name": "Café कोण",
                "business_address": "12 MG Road, Pune, MH 411001",
                "country": "India",
            },
            {
                "entity_id": "S1-2",
                "business_name": "",
                "business_address": "",
                "country": "US",
            },
        ]
    )
    source2 = pd.DataFrame(
        [
            {
                "entity_id": "S2-1",
                "business_name": "कोण Cafe",
                "business_address": "12 M G Rd, Pune, MH 411001",
                "country": "India",
            },
            {
                "entity_id": "S2-2",
                "business_name": "Cafe Corner",
                "business_address": "99 MG Road, Pune, MH 411001",
                "country": "India",
            },
        ]
    )
    source3 = pd.DataFrame(
        [
            {
                "entity_id": "S3-1",
                "business_name": "Cafe Corner",
                "business_address": "12 MG Road, Pune, MH 411001",
                "country": "India",
            }
        ]
    )
    return source1, source2, source3


class CandidateAndRecordTests(unittest.TestCase):
    def test_disk_backed_temporary_exact_candidates_keep_complete_selected_groups(self):
        source1, source2, source3 = source_frames()
        source2 = source2.copy()
        source2.loc[0, "business_name"] = source1.loc[0, "business_name"]
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source_paths = []
            for name, frame in zip(("source2", "source3"), (source2, source3)):
                path = directory / f"{name}.tsv"
                frame.to_csv(path, sep="\t", index=False)
                source_paths.append(path)
            source1_path = directory / "source1.tsv"
            source1.to_csv(source1_path, sep="\t", index=False)
            index = directory / "exact.sqlite"
            temp_exact_candidates.build_index(index, source_paths, rebuild=False)
            output = directory / "candidates.tsv"
            candidates, stats = temp_exact_candidates.generate_candidates(
                index_path=index,
                source1_path=source1_path,
                source1_ids=pd.Series(["S1-1", "S1-2"], dtype="string"),
                output_path=output,
                max_pairs=10,
            )
            self.assertTrue(output.is_file())
            self.assertEqual(stats["selected_source1_entities"], 2)
            self.assertEqual(set(candidates["source1_entity_id"]), {"S1-1"})
            self.assertTrue((candidates["rank"] >= 1).all())

    def test_candidate_contract_rejects_duplicate_and_preserves_metadata(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_candidate_pairs(
                pd.DataFrame(
                    {
                        "source1_entity_id": ["S1-1", "S1-1"],
                        "candidate_entity_id": ["S2-1", "S2-1"],
                    }
                )
            )
        candidate_input = validate_candidate_pairs(
            pd.DataFrame(
                {
                    "source1_entity_id": ["S1-1"],
                    "candidate_entity_id": ["S2-1"],
                    "score": [0.9],
                    "rank": [1],
                }
            )
        )
        self.assertEqual(candidate_input.pairs.columns.tolist(), ["source1_entity_id", "candidate_entity_id"])
        self.assertEqual(candidate_input.retrieval_metadata.columns.tolist(), ["score", "rank"])
        with self.assertRaisesRegex(ValueError, "S1-"):
            validate_candidate_pairs(
                pd.DataFrame({"source1_entity_id": ["S2-1"], "candidate_entity_id": ["S2-1"]})
            )

    def test_dataframe_adapter_batches_consistently_and_rejects_missing_records(self):
        source1, source2, source3 = source_frames()
        adapter = DataFrameRecordAdapter(source1, source2, source3)
        candidates = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-1"],
                "candidate_entity_id": ["S2-1", "S3-1"],
                "rank": [1, 2],
            }
        )
        small = list(adapter.iter_joined_batches(candidates, batch_size=1))
        large = list(adapter.iter_joined_batches(candidates, batch_size=10))
        assert_frame_equal(pd.concat([batch[0] for batch in small], ignore_index=True), large[0][0])
        self.assertEqual(pd.concat([batch[1] for batch in small], ignore_index=True)["rank"].tolist(), [1, 2])
        with self.assertRaises(KeyError):
            list(
                adapter.iter_joined_batches(
                    pd.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-404"]})
                )
            )

    def test_sqlite_store_joins_without_source_dataframe_rescans(self):
        source1, source2, source3 = source_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            paths = []
            for name, frame in zip(("s1", "s2", "s3"), (source1, source2, source3)):
                path = directory / f"{name}.tsv"
                frame.to_csv(path, sep="\t", index=False)
                paths.append(path)
            store = SQLiteRecordStore(directory / "records.sqlite")
            store.build(paths)
            candidates = pd.DataFrame(
                {"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"], "rank": [1]}
            )
            joined, metadata = next(store.iter_joined_batches(candidates, batch_size=1))
            self.assertEqual(joined.loc[0, "source1_business_name"], "Café कोण")
            self.assertEqual(metadata.loc[0, "rank"], 1)


class FeatureScoringDecisionTests(unittest.TestCase):
    def test_frozen_manifest_csv_and_shared_evaluator_known_answer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            train = directory / "train_ids.csv"
            validation = directory / "val_ids.csv"
            pd.DataFrame({"entity_id": ["S1-1"]}).to_csv(train, index=False)
            pd.DataFrame({"entity_id": ["S1-2"]}).to_csv(validation, index=False)
            import hashlib

            manifest = directory / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "split_creation_date": "fixture",
                        "train_source1_count": 1,
                        "validation_source1_count": 1,
                        "train_id_file_sha256": hashlib.sha256(train.read_bytes()).hexdigest(),
                        "validation_id_file_sha256": hashlib.sha256(validation.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            verified = validate_frozen_split_manifest(manifest, train, validation)
            self.assertEqual(verified["train_source1_count"], 1)
        self.assertAlmostEqual(verify_repository_evaluator_known_answer()["macro_f05"], 17 / 18)

    def test_blank_ground_truth_field_is_retained_as_a_singleton(self):
        selected = select_ground_truth(
            pd.DataFrame(
                {
                    "source1_entity_id": ["S1-1", "S1-2"],
                    "matched_entity_ids": ["S2-1", pd.NA],
                }
            ),
            pd.Series(["S1-1", "S1-2"], dtype="string"),
        )
        self.assertEqual(selected.loc[1, "matched_entity_ids"], "")

    def test_stable_train_sample_spans_country_and_match_count_strata(self):
        frozen_ids = pd.Series([f"S1-{number}" for number in range(1, 7)], dtype="string")
        truth = pd.DataFrame(
            {
                "source1_entity_id": frozen_ids,
                "matched_entity_ids": ["", "S2-2", "S2-3,S3-3", "", "S2-5", "S2-6,S3-6"],
            }
        )
        records = pd.DataFrame(
            {
                "entity_id": frozen_ids,
                "country": ["India", "India", "India", "US", "US", "US"],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source1.tsv"
            records.to_csv(path, sep="\t", index=False)
            selected, metadata = select_stable_stratified_train_ids(
                frozen_train_ids=frozen_ids, ground_truth=truth, source1_path=path, sample_size=6, seed=31415
            )
            repeated, repeated_metadata = select_stable_stratified_train_ids(
                frozen_train_ids=frozen_ids, ground_truth=truth, source1_path=path, sample_size=6, seed=31415
            )
        self.assertEqual(selected.tolist(), repeated.tolist())
        self.assertEqual(metadata["selected_source1_ids_sha256"], repeated_metadata["selected_source1_ids_sha256"])
        self.assertEqual(metadata["selected_match_count_groups"], {"0": 2, "1": 2, "2+": 2})
        self.assertEqual(metadata["selected_true_match_pair_count"], 6)

    def test_owned_module_adapters_preserve_existing_public_classes(self):
        self.assertIs(PairwiseFeatureExtractor, OwnedPairwiseFeatureExtractor)
        self.assertIs(DeterministicScorer, OwnedDeterministicScorer)

    def test_unicode_abbreviation_and_reordered_tokens_retain_distinct_evidence(self):
        self.assertEqual(normalize_for_matching("Café कोण"), "café कोण")
        self.assertEqual(token_set("कोण Cafe"), {"कोण", "cafe"})
        self.assertGreater(jaro_winkler_similarity("Acme Corporation", "Acme Corp"), 0.8)
        self.assertEqual(token_set("Cafe Corner"), token_set("Corner Cafe"))
        self.assertGreater(jaro_winkler_similarity("Cafe Corner", "Corner Cafe"), 0.6)

    def test_compiled_distance_path_preserves_reference_values(self):
        first, second = "martha", "marhta"
        self.assertEqual(
            levenshtein_distance_normalized(first, second),
            _python_levenshtein_distance_normalized(first, second),
        )
        self.assertAlmostEqual(
            jaro_winkler_similarity_normalized(first, second),
            _python_jaro_winkler_similarity_normalized(first, second),
            places=12,
        )

    def _joined_pairs(self):
        source1, source2, source3 = source_frames()
        adapter = DataFrameRecordAdapter(source1, source2, source3)
        candidates = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-1", "S1-2"],
                "candidate_entity_id": ["S2-1", "S2-2", "S3-1"],
            }
        )
        return next(adapter.iter_joined_batches(candidates, batch_size=10))[0]

    def test_features_preserve_unicode_and_do_not_reward_missing_fields(self):
        joined = self._joined_pairs()
        extractor = PairwiseFeatureExtractor().fit(joined.iloc[:2])
        features = extractor.transform(joined)
        self.assertGreater(features.loc[0, "name_devanagari_script_match"], 0)
        self.assertEqual(features.loc[2, "name_exact"], 0)
        self.assertEqual(features.loc[2, "address_exact"], 0)
        self.assertEqual(features.loc[1, "numeric_disagreement"], 1)
        self.assertEqual(features.loc[0, "postal_match"], 1)

    def test_features_are_consistent_across_batches_and_persistence(self):
        joined = self._joined_pairs()
        extractor = PairwiseFeatureExtractor().fit(joined.iloc[:2])
        all_features = extractor.transform(joined)
        batched = pd.concat([extractor.transform(joined.iloc[:1]), extractor.transform(joined.iloc[1:])], ignore_index=True)
        assert_frame_equal(all_features, batched)
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "features.pkl"
            extractor.save(artifact)
            restored = PairwiseFeatureExtractor.load(artifact)
            assert_frame_equal(all_features, restored.transform(joined))
            self.assertTrue(artifact.with_suffix(".pkl.json").is_file())

    def test_disk_backed_normalized_corpus_fit_preserves_feature_values(self):
        joined = self._joined_pairs()
        direct = PairwiseFeatureExtractor().fit(joined)
        names = pd.concat([joined["source1_business_name"], joined["candidate_business_name"]]).map(normalize_for_matching)
        addresses = pd.concat([joined["source1_business_address"], joined["candidate_business_address"]]).map(normalize_for_matching)
        disk_style = PairwiseFeatureExtractor().fit_from_normalized_corpora(dict.fromkeys(names), dict.fromkeys(addresses))
        assert_frame_equal(direct.transform(joined), disk_style.transform(joined))

    def test_unique_tfidf_transform_matches_direct_aligned_cosine(self):
        vectorizer = TfidfVectorizer(analyzer="char_wb", lowercase=False).fit(["acme", "acme cafe", "other"])
        left, right = ["acme", "acme", "other"], ["acme cafe", "acme cafe", "other"]
        extractor = PairwiseFeatureExtractor()
        optimized = extractor._tfidf_cosine(vectorizer, left, right)
        direct = (vectorizer.transform(left).multiply(vectorizer.transform(right))).sum(axis=1).A1
        self.assertTrue((abs(optimized - direct) < 1e-12).all())

    def test_raw_scoring_and_multi_match_decision_preserve_empty_entities(self):
        joined = self._joined_pairs()
        features = PairwiseFeatureExtractor().fit(joined.iloc[:2]).transform(joined)
        scores = DeterministicScorer().score(features)
        self.assertIn("raw_match_score", scores.columns)
        self.assertNotIn("match_probability", scores.columns)
        decisions = ThresholdDecisionLayer(threshold=0.0).decide(scores.iloc[:2], ["S1-1", "S1-2", "S1-3"])
        self.assertEqual(decisions.loc[0, "matched_entity_ids"], "S2-1,S2-2")
        self.assertEqual(decisions.loc[1, "matched_entity_ids"], "")
        self.assertEqual(decisions.loc[2, "matched_entity_ids"], "")

    def test_bounded_experiment_writes_probability_contract_on_fixture_only(self):
        source1, source2, source3 = source_frames()
        adapter = DataFrameRecordAdapter(source1, source2, source3)
        train_candidates = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-1"],
                "candidate_entity_id": ["S2-1", "S2-2"],
            }
        )
        validation_candidates = pd.DataFrame(
            {"source1_entity_id": ["S1-2"], "candidate_entity_id": ["S3-1"]}
        )
        ground_truth = pd.DataFrame(
            {
                "source1_entity_id": ["S1-1", "S1-2"],
                "matched_entity_ids": ["S2-1", ""],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = run_bounded_experiment(
                train_ids=pd.Series(["S1-1"], dtype="string"),
                validation_ids=pd.Series(["S1-2"], dtype="string"),
                train_candidates=train_candidates,
                validation_candidates=validation_candidates,
                ground_truth=ground_truth,
                source1_records=source1,
                record_adapter=adapter,
                run_dir=temp_dir,
                config=BoundedExperimentConfig(max_train_pairs=10, max_validation_pairs=10),
                experiment_id="FIXTURE-ONLY",
            )
            probability_scores = pd.read_csv(Path(temp_dir) / "validation_logistic_scores.tsv", sep="\t")
            self.assertEqual(
                probability_scores.columns.tolist(),
                ["source1_entity_id", "candidate_entity_id", "match_probability"],
            )
            self.assertEqual(report["status"], "local_preflight_not_official_until_mudit_confirms_evaluator")
            self.assertEqual(report["validation"]["candidate_recall"], 1.0)

    def test_streaming_scores_preserve_groups_zero_candidates_and_evaluator_contract(self):
        source1, source2, source3 = source_frames()
        pairs = pd.DataFrame(
            {
                # Deliberately interleave the input: the spool must reassemble
                # complete S1 groups before bounded scoring.
                "source1_entity_id": ["S1-2", "S1-1", "S1-1"],
                "candidate_entity_id": ["S3-1", "S2-1", "S2-2"],
            }
        )
        adapter = DataFrameRecordAdapter(source1, source2, source3)
        joined = next(adapter.iter_joined_batches(pairs, batch_size=10))[0]
        extractor = PairwiseFeatureExtractor().fit(joined)
        features = extractor.transform(joined)
        model = fit_logistic_regression(extractor.transform(joined), pd.Series([0, 1, 0]), extractor.feature_order)
        expected = predict_match_probabilities(model, features, extractor.feature_order)
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            source_paths = []
            for name, frame in zip(("s1", "s2", "s3"), (source1, source2, source3)):
                path = directory / f"{name}.tsv"
                frame.to_csv(path, sep="\t", index=False)
                source_paths.append(path)
            store = SQLiteRecordStore(directory / "records.sqlite")
            store.build(source_paths)
            candidates = directory / "candidates.tsv"
            pairs.to_csv(candidates, sep="\t", index=False)
            source_ids = directory / "ids.csv"
            pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"]}).to_csv(source_ids, index=False)
            feature_artifact = directory / "features.pkl"
            model_artifact = directory / "model.pkl"
            extractor.save(feature_artifact)
            with model_artifact.open("wb") as handle:
                pickle.dump(model, handle)
            spool = CandidatePairSpool(directory / "candidate_spool.sqlite")
            metadata = spool.build(candidates, source_ids)
            self.assertEqual(metadata["zero_candidate_source1_count"], 1)
            batches = list(spool.iter_group_batches(2))
            self.assertEqual([len(batch) for _, batch in batches], [2, 1])
            state = score_logistic_stream(
                spool=spool,
                record_store=store,
                extractor=extractor,
                model=model,
                run_dir=directory / "run",
                config=StreamingConfig(pair_batch_size=2),
                model_path=model_artifact,
                feature_artifact_path=feature_artifact,
            )
            self.assertTrue(state["complete"])
            resumed = score_logistic_stream(
                spool=spool,
                record_store=store,
                extractor=extractor,
                model=model,
                run_dir=directory / "run",
                config=StreamingConfig(pair_batch_size=2),
                model_path=model_artifact,
                feature_artifact_path=feature_artifact,
            )
            self.assertEqual(resumed["newly_scored_pair_count"], 0)
            score_path = directory / "scores.tsv"
            output = assemble_score_output(directory / "run", score_path)
            self.assertEqual(output["columns"], ["source1_entity_id", "candidate_entity_id", "match_probability"])
            actual = pd.read_csv(score_path, sep="\t").sort_values(["source1_entity_id", "candidate_entity_id"])
            direct = pd.DataFrame(
                {"source1_entity_id": pairs["source1_entity_id"], "candidate_entity_id": pairs["candidate_entity_id"], "match_probability": expected}
            ).sort_values(["source1_entity_id", "candidate_entity_id"])
            self.assertTrue(np.allclose(actual["match_probability"], direct["match_probability"], rtol=1e-12, atol=1e-12))
            decisions = build_threshold_decisions(
                score_path=score_path,
                source1_ids_path=source_ids,
                output_path=directory / "decisions.tsv",
                threshold=0.5,
                working_database=directory / "decisions.sqlite",
            )
            self.assertEqual(decisions["source1_entity_count"], 3)
            decision_frame = pd.read_csv(directory / "decisions.tsv", sep="\t", dtype="string", keep_default_na=False)
            self.assertEqual(decision_frame.loc[2, "matched_entity_ids"], "")
            truth = directory / "truth.tsv"
            pd.DataFrame(
                {"source1_entity_id": ["S1-1", "S1-2", "S1-3"], "matched_entity_ids": ["S2-1", "", ""]}
            ).to_csv(truth, sep="\t", index=False)
            metrics = evaluate_with_shared_evaluator(
                ground_truth_path=truth, source1_ids_path=source_ids, decisions_path=directory / "decisions.tsv"
            )
            self.assertEqual(metrics["total_entities"], 3)
            errors = compute_disk_backed_error_counts(
                spool=spool, ground_truth_path=truth, decision_database=directory / "decisions.sqlite"
            )
            self.assertEqual(errors["blocking_misses"] + errors["matcher_rejections"], errors["pairwise_fn"])
            self.assertEqual(errors["pairwise_tp"] + errors["pairwise_fp"], decisions["accepted_pair_count"])

    def test_candidate_spool_streams_gzip_and_applies_explicit_rank_cutoff(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            candidates = directory / "candidates.tsv.gz"
            with gzip.open(candidates, "wt", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["source1_entity_id", "candidate_entity_id", "score", "rank"],
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-1", "score": "1", "rank": "1"},
                        {"source1_entity_id": "S1-1", "candidate_entity_id": "S3-1", "score": "0.9", "rank": "2"},
                        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-2", "score": "0.8", "rank": "1"},
                    ]
                )
            source_ids = directory / "ids.csv"
            pd.DataFrame({"entity_id": ["S1-1", "S1-2", "S1-3"]}).to_csv(source_ids, index=False)
            spool = CandidatePairSpool(directory / "candidate_spool.sqlite")
            metadata = spool.build(candidates, source_ids, max_rank=1)
            self.assertEqual(metadata["candidate_pair_count"], 2)
            self.assertEqual(metadata["zero_candidate_source1_count"], 1)
            self.assertEqual(metadata["max_rank"], 1)
            batch = next(spool.iter_group_batches(10))[1]
            self.assertEqual(batch["candidate_entity_id"].tolist(), ["S2-1", "S2-2"])


if __name__ == "__main__":
    unittest.main()
