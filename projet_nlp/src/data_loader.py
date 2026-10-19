"""
data_loader.py
--------------
Chargement et validation des données brutes (CSV).

Fonctions
---------
- charger_csv(chemin, sep=",") → pd.DataFrame
- charger_donnees_test(chemin=None) → pd.DataFrame
"""

import pandas as pd
from .config import CHEMIN_TEST


def charger_csv(chemin: str, sep: str = ",") -> pd.DataFrame:
    """
    Charge un fichier CSV et retourne un DataFrame pandas.

    Paramètres
    ----------
    chemin : str
        Chemin absolu ou relatif vers le fichier CSV.
    sep : str, optionnel
        Séparateur de colonnes (défaut : ',').

    Retourne
    --------
    pd.DataFrame
        Données chargées.

    Lève
    ----
    FileNotFoundError
        Si le fichier est introuvable.
    """
    try:
        df = pd.read_csv(chemin, sep=sep)
        print(f"[Chargement] {chemin} → {df.shape[0]} lignes, {df.shape[1]} colonnes.")
        return df
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fichier introuvable : {chemin}"
        ) from exc


def charger_donnees_test(chemin: str = None) -> pd.DataFrame:
    """
    Charge le jeu de test.

    Paramètres
    ----------
    chemin : str, optionnel
        Chemin vers le CSV de test. Si None, utilise CHEMIN_TEST défini
        dans config.py.

    Retourne
    --------
    pd.DataFrame
        Jeu de test avec colonnes : id, keyword, location, text.
    """
    chemin = chemin or CHEMIN_TEST
    df = charger_csv(chemin)
    colonnes_attendues = {"id", "text"}
    colonnes_manquantes = colonnes_attendues - set(df.columns)
    if colonnes_manquantes:
        raise ValueError(
            f"Colonnes manquantes dans le jeu de test : {colonnes_manquantes}"
        )
    if df.empty:
        raise ValueError("Le jeu de test est vide.")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("Les identifiants doivent être renseignés et uniques.")
    return df
