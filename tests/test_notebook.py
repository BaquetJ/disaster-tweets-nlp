"""Regressions des cellules exploratoires, sans executer le notebook complet."""

import ast
from collections import Counter
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import nbformat
import pandas as pd

from projet_nlp.src.feature_engineering import appliquer_feature_engineering
from projet_nlp.src.training import creer_partitions, groupes_depuis_textes_bruts

ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = nbformat.read(ROOT / "pipeline_ntb.ipynb", as_version=4)
        cls.cells = {
            cell.metadata["exploration_original_cell"]: cell
            for cell in cls.notebook.cells
            if "exploration_original_cell" in cell.metadata
        }

    def definitions(self, *indices):
        namespace = {"pd": pd, "re": re}
        for index in indices:
            tree = ast.parse(self.cells[index].source)
            tree.body = [
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            ]
            exec(compile(tree, f"exploration_cell_{index}", "exec"), namespace)
        return namespace

    def test_original_exploratory_cells_remain_in_order(self):
        ids = [
            cell.metadata["exploration_original_cell"]
            for cell in self.notebook.cells
            if "exploration_original_cell" in cell.metadata
        ]
        self.assertEqual(ids, list(range(117)))
        self.assertGreater(len(self.notebook.cells), 117)
        nbformat.validate(self.notebook)
        for cell in self.notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "<notebook>", "exec")

    def test_abandoned_code_is_preserved_but_inactive(self):
        for index in (90, 92):
            self.assertIn("PISTE NON RETENUE", self.cells[index].source)
            self.assertFalse(ast.parse(self.cells[index].source).body)
        training = self.cells[111].source
        self.assertIn("# class SupervisedModelTrainer:", training)
        self.assertIn("#             rs.fit(self.train_df, self.y_train)", training)
        active = ast.parse(training)
        self.assertFalse(any(isinstance(node, ast.ClassDef) for node in active.body))
        self.assertIn("report = entrainer(", training)

    def test_loading_reserves_holdout_before_exploration(self):
        data = pd.DataFrame({
            "id": range(120),
            "text": [f"tweet {i // 2}" for i in range(120)],
            "target": [(i // 2) % 2 for i in range(120)],
        })
        kaggle = pd.DataFrame({"id": [999], "text": ["unlabeled tweet"]})
        namespace = {
            "pd": pd, "Path": Path, "display": lambda value: None,
            "creer_partitions": creer_partitions,
            "groupes_depuis_textes_bruts": groupes_depuis_textes_bruts,
        }
        with patch.object(pd, "read_csv", side_effect=[kaggle, data]):
            exec(self.cells[21].source, namespace)
        exploration = namespace["df_train"]
        holdout_ids = namespace["ids_holdout_reserves"]
        self.assertFalse(set(exploration["id"]) & holdout_ids)
        self.assertEqual(set(exploration["id"]) | holdout_ids, set(data["id"]))
        self.assertLess(len(exploration), len(data))
        pd.testing.assert_frame_equal(exploration, namespace["df_exploration_brut"])
        groups = namespace["groupes_evaluation"]
        train = namespace["indices_fit"]
        holdout = namespace["indices_holdout"]
        self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[holdout]))
        pd.testing.assert_frame_equal(namespace["df_train_brut"], data)
        source = self.cells[107].source
        self.assertIn("appliquer_feature_engineering(df_exploration_brut)", source)
        self.assertNotIn("appliquer_feature_engineering(df_train_brut)", source)
        self.assertNotIn('normaliser_groupes(features_finales["text"])', "\n".join(
            cell.source for cell in self.notebook.cells if cell.cell_type == "code"
        ))

    def test_digits_include_last_character(self):
        namespace = self.definitions(78)
        data = pd.DataFrame({"text": ["123", "abc9", "9", "", "abc"]})
        result = namespace["compte_nombre"](data)
        self.assertEqual(result["compte_nombre"].tolist(), [3, 1, 1, 0, 0])

    def test_links_include_http_and_https(self):
        namespace = self.definitions(70)
        data = pd.DataFrame({"text": ["https://x", "http://x https://y", "hello"]})
        result = namespace["variable_exacte_lien"](data, "text", seuil=0)
        self.assertEqual(result["nb_liens"].tolist(), [1, 2, 0])

    def test_smileys_require_a_complete_expression(self):
        namespace = self.definitions(72)
        data = pd.DataFrame({"text": [":- ratio", ":)", ":(", ":-)", ";p", "hello"]})
        result = namespace["ajout_smiley"](data)
        self.assertEqual(result["smiley_binaire"].tolist(), [0, 1, 1, 1, 1, 0])

    def test_length_quartiles_use_quantiles_not_mean(self):
        namespace = self.definitions(74)
        data = pd.DataFrame({"text": ["a" * size for size in [1, 2, 3, 4, 100]]})
        result = namespace["ajout_nb_caracteres"](data)
        self.assertEqual(result["nb_caracteres_Q1"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(result["nb_caracteres_Q2"].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(result["nb_caracteres_Q3"].tolist(), [0, 0, 0, 1, 0])
        self.assertEqual(result["nb_caracteres_Q4"].tolist(), [0, 0, 0, 0, 1])

    def test_digit_dummies_read_their_argument(self):
        namespace = self.definitions(80)
        data = pd.DataFrame({"compte_nombre": [0] * 99 + [20]})
        result = namespace["contient_nombre"](data, seuil=2)
        self.assertEqual(result["contient_nombre_rare"].sum(), 1)
        self.assertNotIn("contient_20_nombre", result)

    def test_complete_dates_are_not_double_counted(self):
        namespace = self.definitions(82)
        exec(self.cells[7].source, namespace)
        for text in ["10/07/2015", "10 July 2015", "July 2015", "2015", "07/2015"]:
            with self.subTest(text=text):
                self.assertEqual(
                    len(re.findall(namespace["date_pattern"], text, flags=re.I | re.X)), 1,
                )
        data = pd.DataFrame({"text": ["10/07/2015", "2014 and 2015", "no date"]})
        result = namespace["ajout_nb_dates"](data, max_dates=1)
        self.assertEqual(result["1_dates_text"].tolist(), [1, 0, 0])
        self.assertEqual(result["plus_de_1_dates_text"].tolist(), [0, 1, 0])

    def test_lexicon_respects_words_and_row_order(self):
        namespace = self.definitions(84)
        data = pd.DataFrame(
            {"id": [8, 9, 10], "text": ["fireplace", "FIRE here", "no warning"]},
            index=[5, 2, 4],
        )
        detector = namespace["CatastropheDetection"](data, {"fire"})
        result = detector.get_dummy()
        self.assertEqual(result["catastroph_word_presence"].tolist(), [False, True, False])
        self.assertEqual(result["id"].tolist(), [8, 9, 10])
        self.assertEqual(result.index.tolist(), [5, 2, 4])

    def test_exploratory_retained_features_match_shared_pipeline(self):
        namespace = self.definitions(66, 70, 74, 76, 78, 96, 101)
        exec(self.cells[9].source, namespace)
        exec(self.cells[12].source, namespace)
        exec(self.cells[16].source, namespace)
        from geosky import geo_plug
        namespace["geo_plug"] = geo_plug
        raw = pd.DataFrame({
            "id": [2, 3, 4],
            "text": ["CNN: fire in California 9... https://x", "coool hello", "storm in France"],
        })
        data = raw.copy()
        for name, args in [
            ("variable_exacte_points", ("text",)),
            ("variable_exacte_lien", ("text",)),
            ("ajout_nb_caracteres", ()),
            ("ajout_plusieurs_lettre", ()),
            ("compte_nombre", ()),
        ]:
            data = namespace[name](data, *args)
        data = namespace["LocationMatcher"](
            data, "text", namespace["city_to_country_continent"], namespace["us_states"],
        ).fit_transform()
        data["text"] = namespace["remove_source_prefix"](
            data["text"].str.replace(r"[éê]", "e", regex=True)
        )
        expected = appliquer_feature_engineering(raw)
        pd.testing.assert_frame_equal(data[expected.columns], expected)

    def test_saved_outputs_have_no_errors(self):
        errors = [
            output
            for cell in self.notebook.cells
            for output in cell.get("outputs", [])
            if output.output_type == "error"
        ]
        self.assertFalse(errors)
        counts = Counter(cell.get("id") for cell in self.notebook.cells)
        self.assertTrue(all(count == 1 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
