"""
preprocessor.py
---------------
Construction du ColumnTransformer (TF-IDF + StandardScaler) utilisé
comme étape de pré-traitement dans la pipeline scikit-learn.

Classes
-------
- TextPreprocessorBuilder
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class TextPreprocessorBuilder:
    """
    Construit un ColumnTransformer combinant :
    - TF-IDF sur la colonne texte
    - StandardScaler sur les colonnes numériques

    Attributs
    ---------
    text_column : str
        Nom de la colonne texte.
    numeric_columns : list[str]
        Noms des colonnes numériques.
    max_features : int
        Nombre maximum de features TF-IDF.
    preprocessor : ColumnTransformer ou None
        Construit après appel de build().
    """

    def __init__(
        self,
        text_column: str,
        numeric_columns: list = None,
        max_features: int = 5000,
    ):
        """
        Paramètres
        ----------
        text_column : str
            Nom de la colonne contenant le texte brut des tweets.
        numeric_columns : list[str], optionnel
            Liste des noms de colonnes numériques à normaliser.
        max_features : int, optionnel
            Taille du vocabulaire TF-IDF (défaut : 5000).
        """
        self.text_column = text_column
        self.numeric_columns = list(numeric_columns) if numeric_columns else []
        self.max_features = max_features
        self.preprocessor = None

    def build(self) -> ColumnTransformer:
        """
        Instancie et retourne le ColumnTransformer.

        Retourne
        --------
        ColumnTransformer
            Prêt à être intégré dans une Pipeline scikit-learn.
        """
        tfidf = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
        )

        text_cols = (
            self.text_column
            if isinstance(self.text_column, str)
            else list(self.text_column)
        )

        if self.numeric_columns:
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("tfidf", tfidf, text_cols),
                    ("num", StandardScaler(), self.numeric_columns),
                ],
                remainder="drop",
            )
        else:
            self.preprocessor = ColumnTransformer(
                transformers=[("tfidf", tfidf, text_cols)],
                remainder="drop",
            )

        return self.preprocessor
