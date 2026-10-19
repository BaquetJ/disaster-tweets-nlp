"""
stats_descriptives_test.py
-------------------------
Statistiques descriptives et des graphiques a partir du
jeu de test predit exporte dans output/predictions.csv.

Le script s'inspire de la logique de visualisation presente dans le
notebook, mais l'adapte aux colonnes reellement disponibles dans le
fichier de predictions.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import CHEMIN_PREDICTIONS, DIR_OUTPUT


DEFAULT_OUTPUT_DIR = os.path.join(DIR_OUTPUT, "stat_descriptives_test")
TOP_WORDS_LIMIT = 20


class DescriptiveStatsExporter:
    """Exporte des statistiques descriptives et des visualisations."""

    def __init__(self, input_path: str, output_dir: str) -> None:
        self.input_path = input_path
        self.output_dir = output_dir
        self.tables_dir = os.path.join(output_dir, "tableaux")
        self.plots_dir = os.path.join(output_dir, "graphiques")
        self.df: pd.DataFrame | None = None

        sns.set_theme(style="whitegrid")

    def run(self) -> None:
        self._load_data()
        self._prepare_directories()
        self._export_summary()
        self._export_missing_values()
        self._export_numeric_statistics()
        self._export_binary_statistics()
        self._export_text_statistics()
        self._export_correlations()
        self._export_plots()

    def _load_data(self) -> None:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(
                f"Fichier de predictions introuvable : {self.input_path}"
            )

        self.df = pd.read_csv(self.input_path)

        if self.df.empty:
            raise ValueError("Le fichier de predictions est vide.")

    def _prepare_directories(self) -> None:
        os.makedirs(self.tables_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def _export_summary(self) -> None:
        df = self._require_df()

        summary = {
            "nombre_lignes": len(df),
            "nombre_colonnes": len(df.columns),
            "nombre_colonnes_numeriques": len(self._numeric_columns()),
            "nombre_colonnes_binaires": len(self._binary_columns()),
            "nombre_valeurs_manquantes": int(df.isna().sum().sum()),
        }

        if "target" in df.columns:
            predicted_positive = int(df["target"].fillna(0).sum())
            summary["tweets_predits_catastrophe"] = predicted_positive
            summary["part_catastrophe_pct"] = round(
                predicted_positive / len(df) * 100, 2
            )

        if "proba_catastrophe" in df.columns:
            summary["proba_catastrophe_moyenne"] = round(
                float(df["proba_catastrophe"].mean()), 6
            )
            summary["proba_catastrophe_mediane"] = round(
                float(df["proba_catastrophe"].median()), 6
            )

        summary_df = pd.DataFrame(
            {"indicateur": list(summary.keys()), "valeur": list(summary.values())}
        )
        summary_df.to_csv(
            os.path.join(self.tables_dir, "resume_general.csv"), index=False
        )

    def _export_missing_values(self) -> None:
        df = self._require_df()

        missing_df = pd.DataFrame(
            {
                "colonne": df.columns,
                "nb_manquants": df.isna().sum().values,
                "pct_manquants": (df.isna().mean() * 100).round(2).values,
            }
        ).sort_values(["nb_manquants", "colonne"], ascending=[False, True])

        missing_df.to_csv(
            os.path.join(self.tables_dir, "valeurs_manquantes.csv"), index=False
        )

    def _export_numeric_statistics(self) -> None:
        df = self._require_df()
        numeric_cols = self._numeric_columns()

        if not numeric_cols:
            return

        numeric_stats = df[numeric_cols].describe().transpose().reset_index()
        numeric_stats = numeric_stats.rename(columns={"index": "colonne"})
        numeric_stats.to_csv(
            os.path.join(self.tables_dir, "statistiques_numeriques.csv"),
            index=False,
        )

    def _export_binary_statistics(self) -> None:
        df = self._require_df()

        binary_frames = []
        for column in self._binary_columns():
            counts = df[column].value_counts(dropna=False).sort_index()
            stats_df = pd.DataFrame(
                {
                    "colonne": column,
                    "modalite": counts.index.astype(str),
                    "effectif": counts.values,
                    "pourcentage": (counts.values / len(df) * 100).round(2),
                }
            )
            binary_frames.append(stats_df)

        if binary_frames:
            pd.concat(binary_frames, ignore_index=True).to_csv(
                os.path.join(self.tables_dir, "distributions_binaires.csv"),
                index=False,
            )

    def _export_text_statistics(self) -> None:
        df = self._require_df()

        if "text" not in df.columns:
            return

        text_series = df["text"].fillna("").astype(str)
        text_stats = pd.DataFrame(
            {
                "indicateur": [
                    "longueur_moyenne_texte",
                    "longueur_mediane_texte",
                    "longueur_max_texte",
                    "nombre_textes_vides",
                ],
                "valeur": [
                    round(float(text_series.str.len().mean()), 2),
                    round(float(text_series.str.len().median()), 2),
                    int(text_series.str.len().max()),
                    int((text_series.str.strip() == "").sum()),
                ],
            }
        )
        text_stats.to_csv(
            os.path.join(self.tables_dir, "statistiques_textes.csv"), index=False
        )

        top_words = self._compute_top_words(text_series, limit=TOP_WORDS_LIMIT)
        top_words.to_csv(
            os.path.join(self.tables_dir, "top_mots_textes.csv"), index=False
        )

    def _export_correlations(self) -> None:
        df = self._require_df()
        numeric_cols = self._numeric_columns()

        if len(numeric_cols) < 2:
            return

        corr_df = df[numeric_cols].corr(numeric_only=True)
        corr_df.to_csv(os.path.join(self.tables_dir, "correlations.csv"))

    def _export_plots(self) -> None:
        df = self._require_df()

        if "text" in df.columns:
            self._plot_missing_pie("text")

        self._plot_missing_bar()

        for column in self._binary_columns():
            self._plot_binary(column)

        for column in self._numeric_columns():
            self._plot_numeric(column)

        for column in self._target_comparison_columns():
            self._plot_vs_target(column)

        if "text" in df.columns:
            self._plot_top_words()

        self._plot_target_distribution()
        self._plot_probability_distribution()
        self._plot_correlation_heatmap()

    def _plot_missing_pie(self, column: str) -> None:
        df = self._require_df()
        missing = int(df[column].isna().sum())
        present = len(df) - missing

        plt.figure(figsize=(6, 6))
        plt.pie(
            [present, missing],
            labels=["Present", "Manquant"],
            autopct="%1.1f%%",
            colors=["steelblue", "lightgrey"],
        )
        plt.title(f"Valeurs manquantes : {column}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, f"valeurs_manquantes_{column}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_missing_bar(self) -> None:
        df = self._require_df()
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]

        if missing.empty:
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(x=missing.values, y=missing.index, hue=missing.index, legend=False)
        plt.title("Nombre de valeurs manquantes par colonne")
        plt.xlabel("Nombre de valeurs manquantes")
        plt.ylabel("Colonne")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "valeurs_manquantes_par_colonne.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_binary(self, column: str) -> None:
        df = self._require_df()
        counts = df[column].value_counts(dropna=False).sort_index()
        x_labels = counts.index.astype(str)

        plt.figure(figsize=(6, 4))
        sns.barplot(x=x_labels, y=counts.values, hue=x_labels, palette="coolwarm", legend=False)
        plt.title(f"Distribution de {column}")
        plt.xlabel(column)
        plt.ylabel("Nombre")

        for index, value in enumerate(counts.values):
            plt.text(index, value, str(value), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, f"distribution_{column}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_numeric(self, column: str) -> None:
        df = self._require_df()

        plt.figure(figsize=(8, 5))
        sns.histplot(df[column].dropna(), bins=30, kde=True, color="steelblue")
        plt.title(f"Distribution de {column}")
        plt.xlabel(column)
        plt.ylabel("Nombre de tweets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, f"histogramme_{column}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_vs_target(self, column: str) -> None:
        df = self._require_df()

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=column, hue="target", palette="Set2")
        plt.title(f"{column} selon la target predite")
        plt.xlabel(column)
        plt.ylabel("Nombre de tweets")
        plt.legend(title="Catastrophe")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, f"{column}_selon_target.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_top_words(self) -> None:
        df = self._require_df()
        top_words = self._compute_top_words(df["text"].fillna("").astype(str))

        if top_words.empty:
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_words, x="effectif", y="mot", hue="mot", palette="viridis", legend=False)
        plt.title(f"Top {len(top_words)} mots les plus frequents")
        plt.xlabel("Nombre d'occurrences")
        plt.ylabel("Mot")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "top_mots_textes.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_target_distribution(self) -> None:
        df = self._require_df()

        if "target" not in df.columns:
            return

        counts = df["target"].value_counts().sort_index()
        x_labels = counts.index.astype(str)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=x_labels, y=counts.values, hue=x_labels, palette="Set2", legend=False)
        plt.title("Distribution de la target predite")
        plt.xlabel("target")
        plt.ylabel("Nombre de tweets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "distribution_target.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_probability_distribution(self) -> None:
        df = self._require_df()

        if "proba_catastrophe" not in df.columns:
            return

        plt.figure(figsize=(8, 5))
        sns.histplot(df["proba_catastrophe"].dropna(), bins=30, kde=True, color="darkorange")
        plt.title("Distribution de la probabilite de catastrophe")
        plt.xlabel("proba_catastrophe")
        plt.ylabel("Nombre de tweets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "distribution_proba_catastrophe.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_correlation_heatmap(self) -> None:
        df = self._require_df()
        numeric_cols = self._numeric_columns()

        if len(numeric_cols) < 2:
            return

        corr = df[numeric_cols].corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", square=True)
        plt.title("Matrice de correlation")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.plots_dir, "matrice_correlation.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def _numeric_columns(self) -> list[str]:
        df = self._require_df()
        return [
            column
            for column in df.select_dtypes(include="number").columns
            if column != "id"
        ]

    def _binary_columns(self) -> list[str]:
        df = self._require_df()
        binary_cols = []

        for column in df.columns:
            if column == "id":
                continue

            series = df[column].dropna()
            if series.empty:
                continue

            unique_values = set(series.unique().tolist())
            if unique_values.issubset({0, 1, 0.0, 1.0, True, False}):
                binary_cols.append(column)

        return binary_cols

    def _target_comparison_columns(self) -> list[str]:
        df = self._require_df()

        if "target" not in df.columns:
            return []

        candidate_columns = [
            "loc_dummy",
            "nb_liens",
            "nb_points",
            "is_state",
            "contient_1_liens",
            "plusieurs_lettres_binaire",
        ]
        return [column for column in candidate_columns if column in df.columns]

    def _compute_top_words(
        self,
        text_series: pd.Series,
        limit: int = TOP_WORDS_LIMIT,
    ) -> pd.DataFrame:
        tokens = []
        for text in text_series:
            cleaned_text = re.sub(r"https?://\S+|www\.\S+", " ", text.lower())
            words = re.findall(r"[a-zA-Z']+", cleaned_text)
            tokens.extend(
                word
                for word in words
                if len(word) > 2 and word not in ENGLISH_STOP_WORDS
            )

        top_words = Counter(tokens).most_common(limit)
        return pd.DataFrame(top_words, columns=["mot", "effectif"])

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise RuntimeError("Les donnees ne sont pas chargees.")
        return self.df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genere des statistiques descriptives sur le jeu de test predit "
            "et enregistre les sorties sur disque."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=CHEMIN_PREDICTIONS,
        help="Chemin vers le fichier predictions.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Dossier de sortie pour les tableaux et graphiques.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exporter = DescriptiveStatsExporter(
        input_path=args.input,
        output_dir=args.output_dir,
    )
    exporter.run()

    print("Statistiques descriptives generees avec succes.")
    print(f"Dossier de sortie : {args.output_dir}")


if __name__ == "__main__":
    main()