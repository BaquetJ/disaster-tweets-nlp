"""Variables deterministes partagees par l'entrainement et l'inference."""

import re

import pandas as pd
from geosky import geo_plug

from .config import (
    CITY_TO_COUNTRY_CONTINENT,
    COLS_TO_KEEP,
    SOURCE_KEYWORDS,
    US_STATES,
)


class LocationMatcher:
    """Detection heuristique : villes connues, puis Etats US, puis pays."""

    def __init__(
        self,
        data: pd.DataFrame,
        text_column: str,
        city_dict: dict | None = None,
        us_states: list | None = None,
    ):
        self.data = data.copy()
        self.data["text_clean"] = self.data[text_column].str.lower()
        self.countries = [country.lower() for country in geo_plug.all_CountryNames()]
        self.us_states = US_STATES if us_states is None else us_states
        cities = CITY_TO_COUNTRY_CONTINENT if city_dict is None else city_dict
        self.cities = list(cities)

    def detect_locations(self) -> "LocationMatcher":
        location_found = []
        location_type = []
        for tweet in self.data["text_clean"]:
            found = None
            found_type = None
            for kind, names in (
                ("city", self.cities),
                ("state", self.us_states),
                ("country", self.countries),
            ):
                for name in names:
                    if re.search(rf"\b{re.escape(name)}\b", tweet):
                        found = name
                        found_type = kind
                        break
                if found is not None:
                    break
            location_found.append(found)
            location_type.append(found_type)
        self.data["location_found"] = location_found
        self.data["location_type"] = location_type
        self.data["loc_dummy"] = self.data["location_found"].notna().astype(int)
        return self

    def create_type_dummies(self) -> "LocationMatcher":
        for kind in ("city", "state", "country"):
            self.data[f"is_{kind}"] = self.data["location_type"].eq(kind).astype(int)
        return self

    def fit_transform(self) -> pd.DataFrame:
        return self.detect_locations().create_type_dummies().data


def remove_source_prefix(tweet_series: pd.Series) -> pd.Series:
    """Retire uniquement les prefixes connus, sans tronquer URL et autres deux-points."""
    pattern = r"^\s*(?:" + "|".join(map(re.escape, SOURCE_KEYWORDS)) + r")\s*:\s*"
    return (
        tweet_series.fillna("").astype(str).str.lower()
        .str.replace(pattern, "", regex=True)
    )


def normaliser_texte_modele(textes: pd.Series) -> pd.Series:
    """Normalisation fixe du texte, sans vocabulaire ni seuil appris sur le corpus."""
    return remove_source_prefix(textes.str.replace(r"[éê]", "e", regex=True))


def appliquer_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve index/identifiants et construit des variables independantes du lot."""
    if "text" not in df:
        raise ValueError("La colonne 'text' est obligatoire.")
    if df.empty:
        raise ValueError("Le jeu de tweets est vide.")
    if df["text"].isna().any() or not df["text"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Chaque tweet doit etre un texte non vide.")
    df = df.copy()
    df["nb_liens"] = df["text"].str.count(r"https?://")
    df["contient_1_liens"] = df["nb_liens"].eq(1).astype(int)
    df["nb_caracteres"] = df["text"].str.len()
    df["nb_points"] = df["text"].str.count(r"\.")
    df["compte_nombre"] = df["text"].str.count(r"[0-9]")
    df["plusieurs_lettres_binaire"] = df["text"].map(
        lambda text: int(re.search(r"([^0-9])\1\1", text) is not None)
    )
    df["text"] = normaliser_texte_modele(df["text"])
    df = LocationMatcher(df, "text").fit_transform()
    columns = [column for column in COLS_TO_KEEP if column != "target"]
    if "target" in df:
        columns = ["target"] + columns
    if "id" in df:
        columns = ["id"] + columns
    return df[columns].copy()
