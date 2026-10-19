"""
main.py
-------
Point d'entrée du projet NLP Disaster Tweets.

Usage
-----
    python main.py [OPTIONS]

Options
-------
    --test    : chemin vers le CSV de test
                (défaut : data/test.csv)
    --modele  : chemin vers le fichier .joblib du modèle entraîné
                (défaut : models/best_model.joblib)
    --output  : chemin de sortie pour le CSV de prédictions
                (défaut : output/predictions.csv)

Exemple
-------
    python main.py
    python main.py --test data/test.csv --modele models/best_model.joblib --output output/predictions.csv
"""

import argparse
import sys
import os

# Ajouter le répertoire racine du projet au PYTHONPATH
# pour permettre les imports relatifs depuis src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import executer_pipeline


def parse_arguments() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande.

    Retourne
    --------
    argparse.Namespace
        Objet contenant les arguments parsés.
    """
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Projet NLP — Détection de catastrophes dans les tweets.\n"
            "Applique le modèle entraîné sur le jeu de test et exporte les prédictions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--test",
        type=str,
        default=None,
        metavar="CHEMIN",
        help=(
            "Chemin vers le fichier CSV de test. "
            "Par défaut : data/test.csv (défini dans src/config.py)."
        ),
    )

    parser.add_argument(
        "--modele",
        type=str,
        default=None,
        metavar="CHEMIN",
        help=(
            "Chemin vers le fichier .joblib du modèle entraîné. "
            "Par défaut : models/best_model.joblib (défini dans src/config.py)."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="CHEMIN",
        help=(
            "Chemin de sortie pour le fichier CSV de prédictions. "
            "Par défaut : output/predictions.csv (défini dans src/config.py)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Fonction principale : parse les arguments et lance la pipeline d'inférence.
    """
    args = parse_arguments()

    print("=" * 60)
    print("  Projet NLP — Détection de catastrophes dans les tweets")
    print("=" * 60)

    try:
        df_predictions = executer_pipeline(
            chemin_test=args.test,
            chemin_modele=args.modele,
            chemin_output=args.output,
        )

        print("\n✅ Pipeline exécutée avec succès.")
        print(f"   Aperçu des 5 premières prédictions :")
        print(df_predictions.head().to_string(index=False))

    except FileNotFoundError as e:
        print(f"\n❌ Erreur — Fichier introuvable :\n   {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Erreur — Données invalides :\n   {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue :\n   {e}")
        raise


if __name__ == "__main__":
    main()
