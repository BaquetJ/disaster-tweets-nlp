"""Point d'entree : python -m projet_nlp.train depuis la racine."""

import argparse

from .src.training import entrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement et evaluation groupes, sans fuite du holdout.")
    parser.add_argument("--train", help="CSV d'entrainement (defaut : projet_nlp/data/train.csv)")
    parser.add_argument("--models", help="Dossier des modeles serialises")
    parser.add_argument("--reports", help="Dossier du rapport d'evaluation")
    args = parser.parse_args()
    entrainer(args.train, args.models, args.reports)


if __name__ == "__main__":
    main()
