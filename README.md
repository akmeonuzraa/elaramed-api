# elaraMed-api

API FastAPI qui sert les modèles de Machine Learning entraînés dans [elaraMed](https://github.com/akmeonuzraa/elaraMed), et reproduit exactement la logique de la fonction `predict_specialty()` du notebook `ML.ipynb`.

## Rôle dans l'architecture globale

```
front-elara (Next.js)
   → Supabase Edge Function (elaramed-predict)
      → elaraMed-api (ce repo)  ← charge les modèles .joblib, retourne une prédiction
```

## Routes disponibles

| Route | Méthode | Auth requise | Description |
|---|---|---|---|
| `/health` | GET | Non | Vérifie que le service et les modèles sont chargés |
| `/predict-symptoms` | POST | Oui (`X-API-Key`) | Prédit la spécialité recommandée à partir des symptômes |
| `/predict-mri` | POST | Oui (`X-API-Key`) | Placeholder — module IRM (preuve de concept) pas encore implémenté |

## Installation locale

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Variables d'environnement

Crée un fichier `.env` (jamais commité, voir `.gitignore`) :
```
ELARAMED_API_KEY=<clé secrète générée avec secrets.token_hex(32)>
```

## Lancer en local

```bash
uvicorn main:app --reload
```
Documentation interactive : http://localhost:8000/docs

## Modèles requis (`models/`)

Ces 5 fichiers doivent être copiés depuis `elaraMed/models_export/` :
- `best_model.joblib`
- `tfidf_vectorizer.joblib`
- `label_encoder.joblib`
- `zone_columns.joblib`
- `numeric_ranges.joblib`

⚠️ **Point de vigilance** : l'ordre des features numériques dans `main.py` (`[len(symptoms), severity, age, pain]`) doit correspondre exactement à l'ordre des colonnes utilisé dans `df_num` lors de l'entraînement (`num_symptoms_clean`, `severity_score`, `age_onset`, `pain_intensity`).

## Déploiement

Dockerfile inclus, compatible Render ou Google Cloud Run.

```bash
gcloud run deploy elaramed-api --source . --region europe-west1 --allow-unauthenticated --set-env-vars ELARAMED_API_KEY=<clé>
```

## Sécurité

- Toutes les routes de prédiction exigent le header `X-API-Key`, qui doit correspondre à `ELARAMED_API_KEY`
- CORS restreint aux origines connues (Supabase)
- Aucun secret dans le code — tout passe par variables d'environnement

## TODO
- [ ] Module IRM : CNN preuve de concept (transfer learning, dataset public)
- [ ] Collecter `body_zone`, `age`, `pain`, `severity` côté frontend (actuellement valeurs par défaut envoyées par l'Edge Function)
