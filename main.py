"""
elaraMed-api — API de prédiction de spécialité médicale
Reproduit exactement la logique de predict_specialty() écrite dans ML.ipynb.
"""

import os
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# ------------------------------------------------------------------
# Chargement des artefacts au démarrage (une seule fois, pas à chaque requête)
# ------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

try:
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    zone_columns = joblib.load(os.path.join(MODELS_DIR, "zone_columns.joblib"))
    numeric_ranges = joblib.load(os.path.join(MODELS_DIR, "numeric_ranges.joblib"))
except FileNotFoundError as e:
    raise RuntimeError(
        f"Fichier modèle manquant : {e}. "
        f"Vérifie que les 5 .joblib sont bien dans {MODELS_DIR}"
    )

mins = numeric_ranges["mins"]
maxs = numeric_ranges["maxs"]
# Ordre vérifié : df_num.columns = ['num_symptoms_clean', 'severity_score',
# 'age_onset', 'pain_intensity'] → correspond à [len(symptoms), severity, age, pain]

app = FastAPI(title="elaraMed API", version="1.0")

# CORS — n'autorise que les origines connues (Supabase + ton domaine Netlify)
# En local, tu peux temporairement ajouter "http://localhost:3000" pendant les tests.
ALLOWED_ORIGINS = [
    "https://rinmmiqjjwbzldhsrxcs.supabase.co",
    # "https://ton-site.netlify.app",  # décommente et remplace une fois connu
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Authentification interne — seul Supabase (qui connaît le secret) peut appeler l'API
# ------------------------------------------------------------------
INTERNAL_API_KEY = os.environ.get("dad8ce6efe1485043bdb64ae99667907f132b4affbd9611944a8b248ee5f2211")
if not INTERNAL_API_KEY:
    raise RuntimeError(
        "La variable d'environnement ELARAMED_API_KEY n'est pas définie. "
        "Génère une clé secrète et configure-la sur Render + dans les secrets Supabase."
    )


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide.")
    return True


# ------------------------------------------------------------------
# Schéma de la requête
# ------------------------------------------------------------------
class SymptomRequest(BaseModel):
    symptoms: List[str] = Field(..., description="Liste de termes HPO en anglais")
    body_zone: str = Field(..., description="Zone corporelle, ex: 'knee', 'lower limb'")
    age: float = Field(..., ge=0, le=120)
    pain: float = Field(..., ge=0, le=10, description="Intensité de la douleur (0-10)")
    severity: float = Field(..., description="Score de sévérité")


class TopSpecialty(BaseModel):
    specialty: str
    probability: float


class SymptomResponse(BaseModel):
    specialty: str
    top3: List[TopSpecialty]


# ------------------------------------------------------------------
# Route de santé (utile pour vérifier que le service tourne)
# ------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_type": type(best_model).__name__}


# ------------------------------------------------------------------
# Route principale — reproduit predict_specialty() du notebook
# ------------------------------------------------------------------
@app.post("/predict-symptoms", response_model=SymptomResponse)
def predict_symptoms(req: SymptomRequest, authorized: bool = Depends(verify_api_key)):
    if not req.symptoms:
        raise HTTPException(status_code=400, detail="La liste de symptômes est vide.")

    # --- Transformation texte (symptômes → TF-IDF) ---
    hpo_str = " ".join(req.symptoms)
    x_tfidf = tfidf.transform([hpo_str])

    # --- Encodage one-hot de la zone corporelle ---
    zone_row = {col: 0.0 for col in zone_columns}
    zone_key = f"zone_{req.body_zone}"
    if zone_key in zone_row:
        zone_row[zone_key] = 1.0
    x_zone = csr_matrix([[zone_row[c] for c in zone_columns]])

    # --- Normalisation des variables numériques (même formule que le notebook) ---
    raw_num = np.array([[len(req.symptoms), req.severity, req.age, req.pain]])
    x_num_n = csr_matrix((raw_num - mins) / (maxs - mins + 1e-9))

    # --- Construction du vecteur final patient ---
    x_patient = hstack([x_tfidf, x_zone, x_num_n])

    # --- Prédiction ---
    pred_idx = best_model.predict(x_patient)[0]
    pred_spec = le.inverse_transform([pred_idx])[0]

    if hasattr(best_model, "predict_proba"):
        proba = best_model.predict_proba(x_patient)[0]
        top3 = sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]
    else:
        top3 = [(pred_spec, 1.0)]

    return SymptomResponse(
        specialty=pred_spec,
        top3=[TopSpecialty(specialty=s, probability=float(p)) for s, p in top3],
    )


# ------------------------------------------------------------------
# Route IRM — placeholder, à remplacer par le vrai CNN preuve de concept
# ------------------------------------------------------------------
@app.post("/predict-mri")
def predict_mri(authorized: bool = Depends(verify_api_key)):
    raise HTTPException(
        status_code=501,
        detail="Module IRM pas encore branché — étape suivante.",
    )