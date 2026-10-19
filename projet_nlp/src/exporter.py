"""
exporter.py
-----------
Sauvegarde des résultats de prédiction sur disque.

Fonctions
---------
- sauvegarder_predictions(df, chemin=None)
"""

import os
import pandas as pd
from .config import CHEMIN_PREDICTIONS


def sauvegarder_predictions(df: pd.DataFrame, chemin: str = None) -> None:
    """
    Exporte le DataFrame de prédictions dans un fichier CSV.

    Le fichier produit contient exactement les colonnes 'id' et 'target',
    ce qui le rend directement utilisable pour une soumission Kaggle.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame contenant au moins les colonnes 'id' et 'target'.
    chemin : str, optionnel
        Chemin de destination. Si None, utilise CHEMIN_PREDICTIONS
        défini dans config.py.

    Lève
    ----
    ValueError
        Si la colonne 'target' est absente du DataFrame.
    """
    chemin = chemin or CHEMIN_PREDICTIONS

    if not {"id", "target"}.issubset(df.columns):
        raise ValueError("Les colonnes 'id' et 'target' sont obligatoires.")
    if df.empty or df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Les prédictions doivent avoir des identifiants uniques et non vides.")
    if not df["target"].isin([0, 1]).all():
        raise ValueError("Les prédictions doivent être binaires (0 ou 1).")

    # Créer le répertoire de destination si nécessaire
    os.makedirs(os.path.dirname(os.path.abspath(chemin)), exist_ok=True)

    df[["id", "target"]].to_csv(chemin, index=False)
    print(f"[Export] Prédictions sauvegardées → {chemin} ({len(df)} lignes)")
