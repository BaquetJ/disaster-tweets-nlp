import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from flask_app.app import create_app
from projet_nlp.src.config import COLS_NUMERIQUES
from projet_nlp.src.data_loader import charger_csv, charger_donnees_test
from projet_nlp.src.exporter import sauvegarder_predictions
from projet_nlp.src.feature_engineering import (
    appliquer_feature_engineering,
    remove_source_prefix,
)
from projet_nlp.src.pipeline import executer_pipeline
from projet_nlp.src.predictor import Predictor
from projet_nlp.src.preprocessor import TextPreprocessorBuilder
from projet_nlp.src.training import (
    bootstrap_intervalles,
    creer_partitions,
    entrainer,
    groupes_depuis_textes_bruts,
    normaliser_groupes,
)


class FeatureTests(unittest.TestCase):
    def test_ids_order_and_index_are_preserved(self):
        data = pd.DataFrame(
            {"id": [10, 20], "text": ["hello there", "fire in California"]},
            index=[9, 3],
        )
        result = appliquer_feature_engineering(data)
        self.assertEqual(result.index.tolist(), [9, 3])
        self.assertEqual(result["id"].tolist(), [10, 20])
        self.assertEqual(result["text"].tolist(), ["hello there", "fire in california"])
        self.assertEqual(data.loc[9, "text"], "hello there")

    def test_features_do_not_depend_on_batch(self):
        tweet = pd.DataFrame({"id": [0], "text": ["CNN: fire https://example.com"]})
        batch = pd.concat([
            tweet, pd.DataFrame({"id": range(1, 21), "text": ["hello"] * 20}),
        ], ignore_index=True)
        single = appliquer_feature_engineering(tweet)
        together = appliquer_feature_engineering(batch).iloc[[0]]
        assert_frame_equal(single, together)
        self.assertEqual(single.iloc[0]["contient_1_liens"], 1)

    def test_urls_and_ordinary_colons_are_preserved(self):
        result = remove_source_prefix(pd.Series([
            "https://example.com", "Time: earthquake", "CNN: earthquake",
        ]))
        self.assertEqual(result.tolist(), [
            "https://example.com", "time: earthquake", "earthquake",
        ])

    def test_counts_and_geography(self):
        data = pd.DataFrame({"text": ["Fire in California 12... https://x http://y coool"]})
        row = appliquer_feature_engineering(data).iloc[0]
        self.assertEqual(row["nb_liens"], 2)
        self.assertEqual(row["contient_1_liens"], 0)
        self.assertEqual(row["compte_nombre"], 2)
        self.assertEqual(row["nb_points"], 3)
        self.assertEqual(row["is_state"], 1)
        self.assertEqual(row["loc_dummy"], 1)
        self.assertEqual(row["plusieurs_lettres_binaire"], 1)

    def test_invalid_text_is_rejected(self):
        for text in [None, "", "   ", 12]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                appliquer_feature_engineering(pd.DataFrame({"text": [text]}))
        for data in [pd.DataFrame({"text": []}), pd.DataFrame({"id": [1]})]:
            with self.assertRaises(ValueError):
                appliquer_feature_engineering(data)


class EvaluationTests(unittest.TestCase):
    def test_raw_groups_match_model_normalization_without_feature_fitting(self):
        raw = pd.DataFrame({"text": [
            "CNN: fire in California", "fire in california",
            "tempête here", "tempete here", "CNN: CNN: fire", "CNN: fire",
        ]})
        expected = normaliser_groupes(appliquer_feature_engineering(raw)["text"])
        actual = groupes_depuis_textes_bruts(raw["text"])
        pd.testing.assert_series_equal(actual, expected)
        self.assertEqual(actual.iloc[0], actual.iloc[1])
        self.assertEqual(actual.iloc[2], actual.iloc[3])
        self.assertNotEqual(actual.iloc[4], actual.iloc[5])
        for text in [None, "", "   ", 12]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                groupes_depuis_textes_bruts(pd.Series([text]))

    def test_training_writes_reproducible_artifacts(self):
        data = pd.DataFrame({
            "id": range(120),
            "text": [
                f"{'earthquake fire danger' if i % 2 else 'happy sunny holiday'} sample{i}"
                for i in range(120)
            ],
            "target": [i % 2 for i in range(120)],
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train.csv"
            data.to_csv(source, index=False)
            events = []

            def split_first(*args):
                events.append("split")
                return creer_partitions(*args)

            def features_after(*args):
                events.append("features")
                return appliquer_feature_engineering(*args)

            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch("projet_nlp.src.training.creer_partitions", side_effect=split_first),
                patch("projet_nlp.src.training.appliquer_feature_engineering", side_effect=features_after),
            ):
                report = entrainer(source, root / "models", root / "reports")
            self.assertEqual(events, ["split", "features"])
            self.assertEqual(report["data"]["overlapping_groups"], 0)
            self.assertEqual(report["data"]["n_train"] + report["data"]["n_holdout"], len(data))
            expected = max(report["cv"], key=lambda key: report["cv"][key]["mean_f1"])
            self.assertEqual(report["selected_model"], expected)
            self.assertTrue((root / "reports" / "evaluation.json").is_file())
            holdout = pd.read_csv(root / "reports" / "holdout_predictions.csv")
            self.assertEqual(len(holdout), report["data"]["n_holdout"])
            self.assertNotIn("text", holdout)
            model = joblib.load(root / "models" / "best_model.joblib")
            self.assertEqual(model.named_steps["pre"].n_features_in_, len(COLS_NUMERIQUES) + 1)
            self.assertEqual(model.named_steps["clf"].class_weight, report["cv"][expected]["best_params"]["clf__class_weight"])

    def test_group_splits_have_no_duplicate_leakage(self):
        X = pd.DataFrame({"text": [f"tweet {i // 2}" for i in range(120)]})
        y = pd.Series([(i // 2) % 2 for i in range(120)])
        groups = normaliser_groupes(X["text"])
        train, holdout, folds = creer_partitions(X, y, groups)
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[holdout]))
        self.assertEqual(set(train) | set(holdout), set(range(len(X))))
        for fit, validation in folds:
            self.assertFalse(set(groups.iloc[train[fit]]) & set(groups.iloc[train[validation]]))
        again = creer_partitions(X, y, groups)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(holdout, again[1])

    def test_group_normalization(self):
        groups = normaliser_groupes(pd.Series([" FIRE  here ", "fire here"]))
        self.assertEqual(groups.nunique(), 1)

    def test_bootstrap_matches_reference_group_resampling(self):
        y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        pred = np.array([0, 1, 1, 0, 0, 1, 0, 0])
        groups = np.repeat(np.arange(4), 2)
        result = bootstrap_intervalles(y, pred, groups, n_bootstrap=100)
        rng = np.random.default_rng(42)
        scores = []
        for _ in range(100):
            rows = np.concatenate([
                np.flatnonzero(groups == group) for group in rng.integers(0, 4, 4)
            ])
            scores.append([
                accuracy_score(y[rows], pred[rows]),
                f1_score(y[rows], pred[rows], zero_division=0),
            ])
        expected = np.quantile(scores, [0.025, 0.975], axis=0)
        for i, name in enumerate(["accuracy", "f1"]):
            np.testing.assert_allclose(
                [result[name]["lower"], result[name]["upper"]], expected[:, i],
            )
        self.assertEqual(result, bootstrap_intervalles(y, pred, groups, n_bootstrap=100))
        self.assertLess(result["f1"]["lower"], result["f1"]["upper"])

    def test_bootstrap_perfect_predictions(self):
        result = bootstrap_intervalles([0, 1] * 10, [0, 1] * 10, range(20), 100)
        self.assertEqual(result["accuracy"]["lower"], 1)
        self.assertEqual(result["f1"]["upper"], 1)

    def test_bootstrap_invalid_input(self):
        for args in [
            ([], [], []), ([0, 1], [0], [1, 2]), ([0, 0], [0, 0], [1, 2]),
            ([0, 1], [0, 1], [1, 1]), ([0, 2], [0, 1], [1, 2]),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                bootstrap_intervalles(*args)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.DataFrame({
            "id": range(6),
            "text": [
                "fire in California", "earthquake destroyed houses", "flood danger",
                "hello sunny day", "love the beach", "happy holiday today",
            ],
        })
        cls.features = appliquer_feature_engineering(cls.raw)
        cls.model = Pipeline([
            ("pre", TextPreprocessorBuilder("text", COLS_NUMERIQUES).build()),
            ("clf", LogisticRegression(random_state=42)),
        ]).fit(cls.features, [1, 1, 1, 0, 0, 0])

    def test_prediction_and_serialization_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            joblib.dump(self.model, path)
            loaded = joblib.load(path)
            assert_frame_equal(
                Predictor(self.model).predire(self.features),
                Predictor(loaded).predire(self.features),
            )
        result = Predictor(self.model).predire_et_formater(self.features)
        self.assertEqual(result.columns.tolist(), ["id", "target"])
        self.assertEqual(result["id"].tolist(), self.raw["id"].tolist())

    def test_single_and_batch_probabilities_agree(self):
        predictor = Predictor(self.model)
        one = predictor.predire(appliquer_feature_engineering(self.raw.iloc[[0]]))
        batch = predictor.predire(self.features)
        self.assertAlmostEqual(one.iloc[0]["proba_catastrophe"], batch.iloc[0]["proba_catastrophe"])

    def test_positive_class_position_is_respected(self):
        class ReversedClasses:
            classes_ = np.array([1, 0])

            def predict(self, data):
                return np.ones(len(data), dtype=int)

            def predict_proba(self, data):
                return np.tile([0.8, 0.2], (len(data), 1))

        result = Predictor(ReversedClasses()).predire(self.features)
        np.testing.assert_allclose(result["proba_catastrophe"], 0.8)

    def test_end_to_end_csv_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "test.csv"
            model = Path(directory) / "model.joblib"
            output = Path(directory) / "predictions.csv"
            self.raw.to_csv(source, index=False)
            joblib.dump(self.model, model)
            with contextlib.redirect_stdout(io.StringIO()):
                result = executer_pipeline(str(source), str(model), str(output))
            exported = pd.read_csv(output)
            self.assertEqual(exported.columns.tolist(), ["id", "target"])
            assert_frame_equal(result, exported)

    def test_export_in_current_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            data = pd.DataFrame({"id": [1], "target": [0], "extra": [123]})
            with contextlib.redirect_stdout(io.StringIO()):
                sauvegarder_predictions(data, str(path))
            self.assertEqual(pd.read_csv(path).columns.tolist(), ["id", "target"])
        with patch("projet_nlp.src.exporter.os.makedirs") as mkdir, patch.object(pd.DataFrame, "to_csv"):
            sauvegarder_predictions(data, "predictions.csv")
            self.assertTrue(Path(mkdir.call_args.args[0]).is_absolute())

    def test_export_rejects_invalid_results(self):
        for data in [
            pd.DataFrame({"target": [1]}),
            pd.DataFrame({"id": [1, 1], "target": [0, 1]}),
            pd.DataFrame({"id": [1], "target": [2]}),
        ]:
            with self.assertRaises(ValueError):
                sauvegarder_predictions(data)

    def test_loader_rejects_duplicate_ids_and_malformed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("id,text\n1,hello\n1,world\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                charger_donnees_test(str(path))
            path.write_text("id,text\n1,hello\n2,too,many,fields\n", encoding="utf-8")
            with self.assertRaises(pd.errors.ParserError):
                charger_csv(str(path))

    def test_web_prediction_matches_shared_pipeline(self):
        app = create_app(self.model)
        app.config["TESTING"] = True
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        response = client.post("/", data={"texte": self.raw.iloc[0]["text"]})
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        expected = Predictor(self.model).predire(self.features).iloc[0]["proba_catastrophe"]
        self.assertIn(f"{expected * 100:.1f}", html)
        self.assertNotIn("Wilson", html)
        self.assertIn("calibration", html)

    def test_web_empty_input(self):
        client = create_app(self.model).test_client()
        response = client.post("/", data={"texte": "  "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Veuillez renseigner", response.get_data(as_text=True))

    def test_web_escapes_user_input(self):
        client = create_app(self.model).test_client()
        response = client.post("/", data={"texte": "<script>alert(1)</script>"})
        self.assertNotIn("<script>", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
