from flask import Flask, render_template, request
import os
import sys
import pandas as pd

# Ensure project package is importable (project root is parent of this flask_app folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Imports from the modular project
from projet_nlp.src.model_loader import charger_modele
from projet_nlp.src.feature_engineering import appliquer_feature_engineering
from projet_nlp.src.predictor import Predictor

def create_app(modele=None) -> Flask:
    app = Flask(__name__)
    predicteur = Predictor(charger_modele() if modele is None else modele)

    @app.route("/", methods=["GET", "POST"])
    def index():
        result = None
        error = None
        status = 200
        if request.method == "POST":
            texte = request.form.get("texte", "").strip()
            if not texte:
                error = "Veuillez renseigner un texte."
                status = 400
            else:
                try:
                    df_in = pd.DataFrame({"id": [0], "text": [texte]})
                    df_feat = appliquer_feature_engineering(df_in)
                    df_pred = predicteur.predire(df_feat)
                    result = {
                        "prediction": int(df_pred.iloc[0]["target"]),
                        "probability": (
                            float(df_pred.iloc[0]["proba_catastrophe"])
                            if "proba_catastrophe" in df_pred else None
                        ),
                    }
                except ValueError:
                    app.logger.exception("Impossible de traiter le tweet.")
                    error = "Le texte fourni ne peut pas être traité."
                    status = 400
        return render_template("index.html", result=result, error=error), status

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
