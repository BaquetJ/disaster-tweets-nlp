"""
predictor.py
------------
Application du modèle chargé sur le jeu de test pré-traité.

Classes
-------
- Predictor
"""

import pandas as pd


class Predictor:
    """
    Applique un modèle scikit-learn sur un DataFrame et retourne les
    prédictions enrichies.

    Attributs
    ---------
    modele : sklearn.pipeline.Pipeline
        Pipeline complet (préprocesseur + classifieur).
    """

    def __init__(self, modele):
        """
        Paramètres
        ----------
        modele : sklearn.pipeline.Pipeline
            Modèle chargé via model_loader.charger_modele().
        """
        self.modele = modele

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Génère les prédictions binaires et, si disponibles, les probabilités
        associées à la classe positive (catastrophe = 1).

        Le DataFrame passé en entrée ne doit PAS contenir la colonne 'target'
        (variable cible absente du jeu de test Kaggle). Si elle est présente
        par erreur, elle est retirée avant l'inférence.

        Colonnes ajoutées au DataFrame retourné :
        - 'target'            : prédiction binaire (0 ou 1)
        - 'proba_catastrophe' : probabilité classe 1 (si predict_proba dispo)

        Paramètres
        ----------
        df : pd.DataFrame
            Jeu de test après feature engineering et alignement des colonnes.

        Retourne
        --------
        pd.DataFrame
            Copie du DataFrame enrichi des colonnes de prédiction.
        """
        df_result = df.copy()

        # Retirer 'target' si présente (elle n'existe pas dans le jeu de test,
        # mais on se protège contre une entrée accidentelle)
        colonnes_features = df_result.drop(columns=["target"], errors="ignore")

        df_result["target"] = self.modele.predict(colonnes_features)

        if hasattr(self.modele, "predict_proba"):
            probas = self.modele.predict_proba(colonnes_features)
            classes = list(self.modele.classes_)
            if 1 not in classes:
                raise ValueError("Le modèle ne contient pas la classe positive 1.")
            df_result["proba_catastrophe"] = probas[:, classes.index(1)]

        return df_result

    def predire_et_formater(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit et retourne uniquement les colonnes nécessaires pour la
        soumission Kaggle : id et target.

        Paramètres
        ----------
        df : pd.DataFrame
            Jeu de test après feature engineering.

        Retourne
        --------
        pd.DataFrame
            DataFrame.
        """
        df_pred = self.predire(df)
        if "id" not in df_pred:
            raise ValueError("La colonne 'id' est obligatoire pour l'export.")
        return df_pred[["id", "target"]].copy()
