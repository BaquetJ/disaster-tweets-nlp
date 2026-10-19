"""
model_loader.py
---------------
Chargement du modèle sérialisé (.joblib) produit par le notebook
d'entraînement.

Fonctions
---------
- charger_modele(chemin=None) → Pipeline scikit-learn
- charger_colonnes_numeriques(chemin=None) → list[str]
"""

import joblib
from .config import CHEMIN_MODELE, CHEMIN_COLS_NUMERIQUES


def charger_modele(chemin: str = None):
    """
    Charge et retourne le meilleur modèle entraîné depuis un fichier .joblib.

    Paramètres
    ----------
    chemin : str, optionnel
        Chemin vers le fichier .joblib du modèle. Si None, utilise
        CHEMIN_MODELE défini dans config.py.

    Retourne
    --------
    sklearn.pipeline.Pipeline
        Pipeline complet (préprocesseur + classifieur).

    Lève
    ----
    FileNotFoundError
        Si le fichier .joblib est introuvable.
    """
    chemin = chemin or CHEMIN_MODELE
    try:
        modele = joblib.load(chemin)
        print(f"[Modèle] Chargé depuis : {chemin}")
        return modele
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Modèle introuvable : {chemin}\n"
            "Veuillez d'abord exécuter le notebook d'entraînement "
            "(pipeline_ntb.ipynb) pour générer le fichier .joblib."
        ) from exc


def charger_colonnes_numeriques(chemin: str = None) -> list:
    """
    Charge la liste des colonnes numériques utilisées lors de l'entraînement.

    Paramètres
    ----------
    chemin : str, optionnel
        Chemin vers le fichier .joblib des colonnes. Si None, utilise
        CHEMIN_COLS_NUMERIQUES défini dans config.py.

    Retourne
    --------
    list[str]
        Liste des noms de colonnes numériques.

    Lève
    ----
    FileNotFoundError
        Si le fichier .joblib est introuvable.
    """
    chemin = chemin or CHEMIN_COLS_NUMERIQUES
    try:
        cols = joblib.load(chemin)
        print(f"[Colonnes numériques] Chargées depuis : {chemin}")
        return cols
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier de colonnes introuvable : {chemin}\n"
            "Veuillez d'abord exécuter le notebook d'entraînement."
        ) from exc
