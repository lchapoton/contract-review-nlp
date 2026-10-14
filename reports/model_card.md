# Model card · Contract clause classifier

> À compléter au fur et à mesure (chiffres issus de `results/`).

## Détails du modèle
- **Base** : `roberta-base` (ou autre, cf. `configs/finetune.yaml`), fine-tuné complet et LoRA
- **Tâche** : classification d'une clause de contrat en 100 types (LEDGAR)
- **Version / date** :
- **Auteur** : Léa

## Usage prévu
- Aide au tri des clauses pendant une due diligence, **avec relecture humaine**.
- Recherche de clauses similaires dans une banque de clauses.

## Usages interdits
- Toute décision juridique automatique sans validation par un juriste.
- Contrats **non anglophones** (le modèle n'a vu que de l'anglais).
- Juger de la validité, de l'équité ou du risque d'une clause : le modèle prédit un *type*, pas un avis.
- Contrats hors du périmètre SEC/US sans évaluation préalable (voir audit).

## Données
- LEDGAR (LexGLUE) : clauses de contrats déposés à la SEC (EDGAR), anglais, ~60k/10k/10k.
- Fortement déséquilibré : ratio max/min des classes = _(voir `results/eda_stats.json`)_.

## Évaluation (test set, utilisé une seule fois)
| Modèle | Params entraînés | Temps | Accuracy | Macro-F1 |
|---|---|---|---|---|
| TF-IDF + LogReg | | | | |
| BiLSTM | | | | |
| RoBERTa full FT | | | | |
| RoBERTa LoRA r=8 | | | | |

Test apparié (McNemar / bootstrap macro-F1) : _à compléter_

## Audit (cf. `results/audit.json`)
- Classes rares vs fréquentes :
- Clauses tronquées :
- Contrefactuels (taux de flip juridiction / partie / genre) :
- Calibration et seuil d'abstention recommandé :

## Limites connues
-
