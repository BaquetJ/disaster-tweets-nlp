"""
pipeline.py
-----------
Orchestration complète de l'inférence :
chargement → feature engineering → prédiction → export.

Fonctions
---------
- executer_pipeline(chemin_test=None, chemin_modele=None, chemin_output=None)
"""

import pandas as pd

from .data_loader import charger_donnees_test
from .feature_engineering import appliquer_feature_engineering
from .model_loader import charger_modele
from .predictor import Predictor
from .exporter import sauvegarder_predictions


def executer_pipeline(
    chemin_test: str = None,
    chemin_modele: str = None,
    chemin_output: str = None,
) -> pd.DataFrame:
    """
    Exécute la pipeline complète d'inférence sur le jeu de test :

    1. Chargement du jeu de test (CSV).
    2. Application du feature engineering (reproduit le notebook).
    3. Chargement du modèle entraîné (.joblib).
    4. Prédiction sur le jeu de test.
    5. Sauvegarde des prédictions (CSV).

    Paramètres
    ----------
    chemin_test : str, optionnel
        Chemin vers le CSV de test. Utilise config.CHEMIN_TEST par défaut.
    chemin_modele : str, optionnel
        Chemin vers le .joblib du modèle. Utilise config.CHEMIN_MODELE par défaut.
    chemin_output : str, optionnel
        Chemin de sortie CSV. Utilise config.CHEMIN_PREDICTIONS par défaut.

    Retourne
    --------
    pd.DataFrame
        DataFrame de prédictions avec colonnes ['id', 'target'].
    """

    # ------------------------------------------------------------------
    # Étape 1 : Chargement des données de test
    # ------------------------------------------------------------------
    print("\n=== Étape 1/5 : Chargement des données ===")
    df_test = charger_donnees_test(chemin_test)

    # ------------------------------------------------------------------
    # Étape 2 : Feature engineering
    # ------------------------------------------------------------------
    print("\n=== Étape 2/5 : Feature engineering ===")
    df_test = appliquer_feature_engineering(df_test)

    # ------------------------------------------------------------------
    # Étape 3 : Chargement du modèle
    # ------------------------------------------------------------------
    print("\n=== Étape 3/5 : Chargement du modèle ===")
    modele = charger_modele(chemin_modele)

    # Les colonnes du DataFrame sont déjà alignées sur COLS_TO_KEEP grâce
    # à appliquer_feature_engineering — aucun alignement supplémentaire
    # n'est nécessaire.

    # ------------------------------------------------------------------
    # Étape 4 : Prédiction
    # ------------------------------------------------------------------
    print("\n=== Étape 4/5 : Prédiction ===")
    predicteur = Predictor(modele)
    df_predictions = predicteur.predire_et_formater(df_test)

    # ------------------------------------------------------------------
    # Étape 5 : Export
    # ------------------------------------------------------------------
    print("\n=== Étape 5/5 : Export des résultats ===")
    sauvegarder_predictions(df_predictions, chemin_output)

    # Résumé
    nb_catastrophes = int(df_predictions["target"].sum())
    nb_total = len(df_predictions)
    print(
        f"\n[Résumé] {nb_catastrophes}/{nb_total} tweets classés comme "
        f"catastrophe réelle ({nb_catastrophes / nb_total * 100:.1f} %)."
    )

    return df_predictions
