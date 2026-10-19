"""Entrainement reproductible et evaluation hors echantillon."""

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .config import COLS_NUMERIQUES, DIR_DATA, DIR_MODELS, DIR_RACINE
from .data_loader import charger_csv
from .feature_engineering import appliquer_feature_engineering, normaliser_texte_modele
from .preprocessor import TextPreprocessorBuilder

SEED = 42


def normaliser_groupes(textes: pd.Series) -> pd.Series:
    """Garde les copies d'un texte dans le meme pli, meme si casse/espace different."""
    return textes.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()


def groupes_depuis_textes_bruts(textes: pd.Series) -> pd.Series:
    """Definit les groupes avant exploration, avec les regles fixes d'inference."""
    if textes.isna().any() or not textes.map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Chaque tweet doit etre un texte non vide.")
    return normaliser_groupes(normaliser_texte_modele(textes))


def creer_partitions(X: pd.DataFrame, y: pd.Series, groupes: pd.Series):
    holdout = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(holdout.split(X, y, groupes))
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    plis = list(cv.split(X.iloc[train_idx], y.iloc[train_idx], groupes.iloc[train_idx]))
    return train_idx, test_idx, plis


def metriques(y, predictions, probabilites) -> dict:
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilites)),
        "average_precision": float(average_precision_score(y, probabilites)),
        "brier": float(brier_score_loss(y, probabilites)),
    }


def bootstrap_intervalles(
    y, predictions, groupes, n_bootstrap: int = 2000, confidence: float = 0.95,
) -> dict:
    """IC percentile conditionnels au modele fixe, avec reechantillonnage des groupes."""
    y = np.asarray(y)
    predictions = np.asarray(predictions)
    groupes = np.asarray(groupes)
    if not (len(y) == len(predictions) == len(groupes)) or len(y) == 0:
        raise ValueError("Les observations, predictions et groupes doivent etre alignes et non vides.")
    if not np.isin(y, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Les observations et predictions doivent etre binaires.")
    if len(np.unique(y)) != 2:
        raise ValueError("Les deux classes sont necessaires pour cet intervalle.")
    if n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("Il faut au moins 100 repetitions et un niveau entre 0 et 1.")
    _, codes = np.unique(groupes, return_inverse=True)
    n_groups = codes.max() + 1
    if n_groups < 2:
        raise ValueError("Au moins deux groupes sont necessaires.")
    # Agreger la matrice de confusion evite de reconstruire les lignes a chaque tirage.
    counts = np.zeros((n_groups, 4), dtype=int)
    np.add.at(counts, (codes, 2 * y.astype(int) + predictions.astype(int)), 1)
    rng = np.random.default_rng(SEED)
    samples = np.empty((n_bootstrap, 2))
    for i in range(n_bootstrap):
        tn, fp, fn, tp = counts[rng.integers(0, n_groups, n_groups)].sum(axis=0)
        samples[i, 0] = (tp + tn) / (tn + fp + fn + tp)
        samples[i, 1] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    tail = (1 - confidence) / 2
    limits = np.quantile(samples, [tail, 1 - tail], axis=0)
    return {
        name: {
            "lower": float(limits[0, i]), "upper": float(limits[1, i]),
            "confidence": confidence, "n_bootstrap": n_bootstrap,
            "method": "percentile_group_bootstrap_fixed_model",
        }
        for i, name in enumerate(("accuracy", "f1"))
    }


def entrainer(chemin_train=None, dossier_modeles=None, dossier_rapport=None) -> dict:
    chemin_train = Path(chemin_train or Path(DIR_DATA) / "train.csv")
    dossier_modeles = Path(dossier_modeles or DIR_MODELS)
    dossier_rapport = Path(dossier_rapport or Path(DIR_RACINE).parent / "reports")
    data = charger_csv(str(chemin_train))
    if not {"id", "text", "target"}.issubset(data):
        raise ValueError("Le CSV d'entrainement doit contenir id, text et target.")
    if not data["target"].isin([0, 1]).all() or data["target"].nunique() != 2:
        raise ValueError("La cible doit contenir exactement les classes 0 et 1.")
    if data["id"].isna().any() or data["id"].duplicated().any():
        raise ValueError("Les identifiants d'entrainement doivent etre uniques et renseignes.")

    y = data["target"]
    groupes = groupes_depuis_textes_bruts(data["text"])
    train_idx, test_idx, plis = creer_partitions(data, y, groupes)
    features = appliquer_feature_engineering(data)
    X = features.drop(columns=["id", "target"])
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    preprocessor = TextPreprocessorBuilder("text", COLS_NUMERIQUES).build()
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED),
            {"clf__C": [0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=1),
            {"clf__max_depth": [10, None], "clf__class_weight": [None, "balanced"]},
        ),
    }
    searches = {}
    for name, (classifier, params) in candidates.items():
        print(f"[Entrainement] {name}", flush=True)
        pipeline = Pipeline([("pre", clone(preprocessor)), ("clf", classifier)])
        search = GridSearchCV(
            pipeline, params, scoring="f1", cv=plis, n_jobs=1,
            error_score="raise", return_train_score=False,
        )
        search.fit(X_train, y_train)
        searches[name] = search

    best_name = max(searches, key=lambda name: searches[name].best_score_)
    evaluation_model = searches[best_name].best_estimator_
    predictions = evaluation_model.predict(X_test)
    probabilities = evaluation_model.predict_proba(X_test)[:, 1]
    baseline = DummyClassifier(strategy="prior", random_state=SEED).fit(X_train, y_train)
    report = {
        "seed": SEED,
        "dataset_sha256": hashlib.sha256(chemin_train.read_bytes()).hexdigest(),
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        },
        "data": {
            "n_total": len(data), "n_train": len(train_idx), "n_holdout": len(test_idx),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().items()},
            "duplicate_normalized_texts": int(groupes.duplicated().sum()),
            "conflicting_label_groups": int(
                pd.DataFrame({"group": groupes, "y": y}).groupby("group")["y"].nunique().gt(1).sum()
            ),
            "overlapping_groups": len(set(groupes.iloc[train_idx]) & set(groupes.iloc[test_idx])),
        },
        "protocol": {
            "holdout": "First fold of seeded 5-fold StratifiedGroupKFold (~20%)",
            "split_timing": "Raw rows partitioned before exploratory feature engineering; notebook exploration restricted to training rows",
            "selection": "Best mean positive-class F1 in 3-fold grouped CV on training only",
            "grouping": "Model-normalized text, lowercased and whitespace-normalized",
            "threshold": 0.5,
            "interval_scope": "Holdout sampling uncertainty conditional on a fixed model; not per-tweet uncertainty",
            "final_model": "Selected configuration refitted on all labeled rows after holdout evaluation",
        },
        "limitations": [
            "Retrospective evaluation: the feature set originated in exploration of this same corpus.",
            "Grouping prevents normalized duplicate leakage, not near-duplicate or shared-event leakage.",
            "Intervals exclude retraining, model selection and distribution shift uncertainty.",
            "Probabilities are not recalibrated; balanced class weights do not guarantee calibration.",
            "No externally verified Kaggle score, temporal validation or operational alerting claim.",
        ],
        "cv": {
            name: {
                "mean_f1": float(search.best_score_),
                "std_f1": float(search.cv_results_["std_test_score"][search.best_index_]),
                "best_params": search.best_params_,
            }
            for name, search in searches.items()
        },
        "selected_model": best_name,
        "holdout_metrics": metriques(y_test, predictions, probabilities),
        "holdout_intervals": bootstrap_intervalles(
            y_test, predictions, groupes.iloc[test_idx],
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "baseline": {
            "strategy": "prior (majority class for predict)",
            "metrics": metriques(
                y_test, baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1],
            ),
        },
    }
    dossier_rapport.mkdir(parents=True, exist_ok=True)
    holdout_predictions = data.iloc[test_idx][["id", "text", "target"]].copy()
    holdout_predictions["prediction"] = predictions
    holdout_predictions["probability"] = probabilities
    # Le rapport public ne recopie pas les textes du jeu de donnees.
    holdout_predictions[["id", "target", "prediction", "probability"]].to_csv(
        dossier_rapport / "holdout_predictions.csv", index=False,
    )
    errors = holdout_predictions[holdout_predictions["target"] != predictions]
    errors[["id", "target", "prediction", "probability"]].to_csv(
        dossier_rapport / "holdout_errors.csv", index=False,
    )

    final_model = clone(evaluation_model).fit(X, y)
    dossier_modeles.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, dossier_modeles / "best_model.joblib")
    joblib.dump(COLS_NUMERIQUES, dossier_modeles / "colonnes_numeriques.joblib")
    joblib.dump(report, dossier_modeles / "metadonnees_entrainement.joblib")
    (dossier_rapport / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(json.dumps(report["holdout_metrics"], indent=2), flush=True)
    return report
