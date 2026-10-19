"""
config.py
---------
Centralise tous les paramètres de configuration du projet :
chemins des fichiers, variables, constantes métier.
"""

import os

# ============================================================
# Chemins
# ============================================================

# Répertoire racine du projet modulaire
DIR_RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Données brutes
DIR_DATA = os.path.join(DIR_RACINE, "data")
CHEMIN_TEST = os.path.join(DIR_DATA, "test.csv")

# Modèle sérialisé (produit par le notebook)
DIR_MODELS = os.path.join(DIR_RACINE, "models")
CHEMIN_MODELE = os.path.join(DIR_MODELS, "best_model.joblib")
CHEMIN_COLS_NUMERIQUES = os.path.join(DIR_MODELS, "colonnes_numeriques.joblib")

# Résultats de prédiction
DIR_OUTPUT = os.path.join(DIR_RACINE, "output")
CHEMIN_PREDICTIONS = os.path.join(DIR_OUTPUT, "predictions.csv")

# ============================================================
# Paramètres de feature engineering
# ============================================================

# États américains (minuscules)
US_STATES = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
]

# Dictionnaire villes → (pays, continent)
CITY_TO_COUNTRY_CONTINENT = {
    "la ronge": ("canada", "north america"),
    "tampa": ("united states", "north america"),
    "colorado springs": ("united states", "north america"),
    "ruby": ("united states", "north america"),
    "manitou": ("united states", "north america"),
}

# Mots-clés sources médias
SOURCE_KEYWORDS = ["dtn", "reuters", "cnn", "bbc", "afp", "ap", "news"]

# ============================================================
# Colonnes retenues pour l'entraînement (liste fixe — notebook)
# ============================================================

# Liste complète telle que définie dans le notebook (cols_to_keep).
# Inclut 'target' (uniquement présente dans le train) et 'text'.
COLS_TO_KEEP = [
    "target",
    "loc_dummy",
    "nb_liens",
    "nb_caracteres",
    "compte_nombre",
    "nb_points",
    "is_state",
    "text",
    "contient_1_liens",
    "plusieurs_lettres_binaire",
]

# Colonnes numériques uniquement (sans 'target' ni 'text')
COLS_NUMERIQUES = [c for c in COLS_TO_KEEP if c not in ("target", "text")]
