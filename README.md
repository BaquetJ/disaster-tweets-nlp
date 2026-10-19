# Disaster Tweets : classification NLP et évaluation reproductible

Projet pédagogique de M2, Université de Lille, réalisé dans le cadre du cours
Big Data. **Il s'agit d'un pipeline NLP local, pas d'un système distribué ni
d'un outil opérationnel de détection de catastrophes.**

Objectif : distinguer les tweets **en anglais** décrivant une catastrophe réelle
(`target=1`) des autres (`target=0`), avec TF-IDF, variables manuelles et
classification supervisée. Une interface Flask illustre l'inférence.

## Résultats mesurés

Sur **1 523 tweets de holdout**, avec sélection du modèle exclusivement par
validation croisée sur les **6 090 autres tweets** :

| Mesure | Régression logistique sélectionnée | Référence majoritaire |
|---|---:|---:|
| F1 de la classe catastrophe | 0,7632 | 0,0000 |
| Précision | 0,7778 | 0,0000 |
| Rappel | 0,7492 | 0,0000 |
| Accuracy | 0,8004 | 0,5706 |
| ROC-AUC | 0,8616 | 0,5000 |
| Average precision | 0,8618 | 0,4294 |
| Brier, plus petit = mieux | 0,1493 | 0,2450 |

- IC bootstrap à 95 % du F1 : **[0,7359 ; 0,7900]**.
- IC bootstrap à 95 % de l'accuracy : **[0,7801 ; 0,8208]**.
- F1 moyen en CV groupée : **0,7404** pour la régression logistique,
  **0,6881** pour la forêt aléatoire. L'écart-type des plis n'est pas un IC.
- Matrice de confusion, lignes = vraie classe, colonnes = classe prédite :
  `[[729, 140], [164, 490]]`.

Le détail vérifiable est dans [reports/evaluation.json](reports/evaluation.json).
Ces résultats sont une **évaluation locale sur ce corpus**, pas un score Kaggle
ni une preuve de généralisation à de nouveaux événements.

## Installation

Environnement vérifié : Python **3.12.10**, scikit-learn **1.9.0**,
pandas **3.0.3**, NumPy **2.4.6**. Les dépendances acceptent des plages bornées ;
les versions ayant produit le rapport sont enregistrées dans celui-ci.
Pour reproduire exactement les valeurs publiées, utiliser ces versions.

Depuis la racine du projet, sous PowerShell :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
```

L'encodage UTF-8 évite les erreurs de console Windows sur les messages en français.
Pour l'inférence seule : [projet_nlp/requirements.txt](projet_nlp/requirements.txt).
Pour Flask : [flask_app/requirements.txt](flask_app/requirements.txt), qui inclut
les dépendances du pipeline.

## Données

Source : [Kaggle — Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data).
Télécharger les fichiers en respectant les conditions de Kaggle et les placer ici :

```text
projet_nlp/
  data/
    train.csv    # 7 613 lignes ; id, keyword, location, text, target
    test.csv     # 3 263 lignes ; mêmes champs sans target
```

Les CSV locaux et les modèles ne seront pas inclus dans le futur dépôt.
L'empreinte SHA-256 du fichier d'entraînement figure dans le rapport.
Les identifiants sont validés ; les lignes CSV mal formées ne sont pas ignorées.
`keyword` et `location` ne sont pas utilisés comme entrées du modèle.

## Reproduire le projet

### 1. Entraîner et évaluer

```powershell
python -m projet_nlp.train
```

Cette commande :

1. Valide les données et réserve environ 20 % des lignes brutes avant le feature
   engineering : premier pli d'un `StratifiedGroupKFold` à cinq plis, graine **42**.
2. Construit huit variables numériques, sans dépendance aux fréquences du lot :
   longueur, liens HTTP/HTTPS, présence d'un seul lien, points, chiffres, répétitions,
   présence d'un lieu et d'un État américain. La normalisation utilisée pour grouper
   les copies est fixe et n'apprend aucun paramètre sur les données.
3. Groupe les textes normalisés identiques pour éviter qu'une copie traverse
   la séparation ou les plis de CV. Les **112 copies normalisées** et les
   **18 groupes à annotations contradictoires** sont conservés et signalés.
4. Compare régression logistique et forêt aléatoire par recherche exhaustive
   d'hyperparamètres en CV groupée à trois plis sur l'entraînement uniquement.
   TF-IDF et standardisation sont ajustés à l'intérieur de chaque pli.
5. Choisit famille et paramètres sur le **F1 moyen de CV**, sans consulter
   les scores du holdout. Le seuil reste fixé à **0,5**.
6. Évalue le modèle sélectionné et une référence majoritaire sur le holdout.
   Calcule 2 000 réplications bootstrap de groupes de textes pour les IC.
7. Réajuste la configuration sélectionnée sur **tout le corpus étiqueté**
   pour produire le modèle final d'inférence.

Les scores et IC décrivent le modèle évalué **avant** ce dernier réajustement.
Les IC sont conditionnels à ce modèle fixe ; ils ne couvrent pas la variabilité
du réentraînement, de la sélection de modèle ou d'un changement de distribution.

Artefacts régénérés :

- [projet_nlp/models/](projet_nlp/models/) : modèle final, colonnes, métadonnées.
- [reports/evaluation.json](reports/evaluation.json) : protocole, métriques, IC,
  paramètres, versions et empreinte des données.
- [reports/holdout_predictions.csv](reports/holdout_predictions.csv) et
  [reports/holdout_errors.csv](reports/holdout_errors.csv) : identifiants, cibles,
  prédictions et probabilités, sans recopier les textes.

Chemins personnalisables :

```powershell
python -m projet_nlp.train --train projet_nlp\data\train.csv --models projet_nlp\models --reports reports
```

### 2. Exploration et évaluation

**[pipeline_ntb.ipynb](pipeline_ntb.ipynb) présente le parcours exploratoire** :
inspection, construction des variables, visualisations et comparaison de modèles.
Les pistes non retenues restent visibles en commentaire, avec leurs raisons et
leurs limites. Les hypothèses sont distinguées des résultats démontrés.

Le holdout est réservé dès le chargement. **L'exploration porte uniquement sur les
6 090 lignes d'entraînement** : aperçus, graphiques, quantiles, modalités rares et
corrélations n'utilisent pas les 1 523 observations réservées. La normalisation
des groupes est identique à celle du CLI et ne dépend pas de statistiques du corpus.

La dernière partie exécute **le même protocole d'évaluation que le CLI**, sur les
mêmes identifiants réservés au départ. Elle présente la référence majoritaire,
le Brier, les IC, la calibration descriptive et des exemples d'erreurs.
Une classe d'entraînement alternative reste en commentaire comme piste non retenue,
car elle sélectionnerait la famille sur le jeu servant aussi à annoncer le score.

Pour tout reproduire, ouvrir le notebook depuis la racine du dépôt, sélectionner
l'environnement installé et exécuter les cellules dans l'ordre. Les données Kaggle
doivent être présentes ; le notebook **réentraîne** et régénère les modèles et rapports.

Pour exécuter uniquement **l'évaluation**, utiliser
`python -m projet_nlp.train`. Les métriques et les intervalles sont enregistrés dans
[reports/evaluation.json](reports/evaluation.json), et les erreurs dans
[reports/holdout_errors.csv](reports/holdout_errors.csv).
Ce corpus a déjà été exploré : cette organisation ne rend pas les choix de variables
passés indépendants des données d'évaluation. Une ablation contrôlée des variables
manuelles n'a pas été réalisée.

### 3. Prédire sur le test Kaggle

```powershell
python -m projet_nlp.main
# Ou avec chemins personnalisés :
python -m projet_nlp.main --test projet_nlp\data\test.csv --output projet_nlp\output\predictions.csv
```

Le CSV de soumission contient **exactement `id,target`**, dans l'ordre du fichier
d'entrée. La cible Kaggle étant inconnue, aucune métrique n'est calculée sur ce CSV.
La classe positive est identifiée par son étiquette, pas par une position supposée.

### 4. Lancer la démonstration Web

```powershell
python flask_app\app.py
```

Ouvrir **http://127.0.0.1:5000**. Le serveur est local, avec debug désactivé.
Il charge le modèle une fois et applique les mêmes transformations que le CLI.
Une saisie vide produit une erreur explicite.

La sortie affichée est une **probabilité estimée de la classe catastrophe**,
pas un taux de certitude. Aucun IC individuel n'est revendiqué :
appliquer Wilson à une probabilité prédite avec `n=1` n'a pas de justification
statistique. La calibration n'est pas garantie, notamment avec pondération
des classes et hors distribution.

Ne charger que des fichiers `.joblib` de confiance : leur désérialisation peut
exécuter du code. L'entraînement local permet de régénérer les modèles plutôt
que de dépendre des binaires historiques.

### 5. Vérifier les régressions

```powershell
python -m unittest discover -s tests -v
```

Les tests ne nécessitent ni modèles préexistants ni téléchargement des données.
Ils couvrent notamment l'alignement identifiants/textes, l'invariance au lot,
le nettoyage des URL, les séparations sans doublons communs, le bootstrap
comparé à une implémentation de référence, la sérialisation, le CSV de soumission,
la parité Web/pipeline et les erreurs de saisie.

## Organisation

```text
pipeline_ntb.ipynb             Exploration de l'entraînement et évaluation sur holdout
requirements.txt              Dépendances complètes (notebook + pipeline + Web)
projet_nlp/
  train.py                    CLI d'entraînement
  main.py                     CLI d'inférence
  src/
    training.py               Séparation, CV, évaluation, bootstrap, réajustement
    feature_engineering.py    Variables partagées entraînement/inférence
    preprocessor.py           TF-IDF + standardisation dans une Pipeline
    data_loader.py            Chargement et validation
    model_loader.py           Chargement des artefacts locaux
    predictor.py              Prédictions et probabilités
    exporter.py               Export strict id,target
    pipeline.py               Orchestration de l'inférence
    config.py                 Chemins et constantes
flask_app/                    Démonstration locale
tests/                        Tests de régression unittest
reports/                      Évaluation locale vérifiable
```

Les étapes exploratoires et les pistes non retenues sont documentées dans le notebook.
La pipeline finale utilise un module dédié sans regroupement de modalités
en fonction des fréquences du lot.
Les tests du notebook vérifient aussi la séparation avant exploration, l'absence
d'identifiants du holdout dans le tableau exploratoire, la conservation des étapes,
le caractère inactif des pistes non retenues et les définitions des variables.

## Limites à présenter honnêtement

- **Indépendance de l'évaluation** : le corpus a déjà servi à des choix
  exploratoires. Une séparation interne, même exécutée avant les graphiques,
  ne remplace pas un nouveau corpus indépendant.
- Le regroupement couvre les textes normalisés identiques, pas tous les
  quasi-doublons ni les événements communs.
- Corpus anglophone historique et labels parfois contradictoires ; pas de
  validation temporelle, géographique ou multilingue.
- Géographie heuristique et incomplète ; ambiguïtés, ironie et usages figurés
  difficiles pour TF-IDF.
- Probabilités non recalibrées ; le Brier n'est pas une garantie de calibration.
- Les exemples d'erreurs les plus confiantes ne sont pas représentatifs.
  Toute amélioration inspirée de ces erreurs exige une nouvelle évaluation.
- Pas de score public Kaggle vérifié, de benchmark transformer ni de promesse
  de performances en production.

Suites pertinentes : nouvelles données indépendantes, séparation par événement
ou date, revue d'annotations, calibration dédiée et comparaison de modèles
sur un protocole fixé à l'avance.

## Préparation du dépôt

[.gitignore](.gitignore) exclut données, modèles, caches et sorties d'inférence.
Le notebook, le code, les tests et les rapports d'évaluation peuvent être versionnés.
L'archive d'origine est conservée à l'extérieur de ce dossier.
**Aucun commit ni push n'est effectué par cette préparation.**

Historique du projet universitaire :
https://gitlab.com/projet-big-data/projet-big-data
