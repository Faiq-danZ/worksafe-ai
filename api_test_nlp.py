"""
Struktur folder:
project/
├── api_test_nlp.py
└── models/nlp_model/
    ├── worksafe_risk_model_best.keras
    └── worksafe_artifacts.json

Langkah Langkah:
python -m venv venv, jika sudah ada aktifkan venv
Aktifkan:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

pip install fastapi uvicorn tensorflow numpy openrouter

Jalankan API:
uvicorn openroute_test:app --reload --host 0.0.0.0 --port 8000

Buka dokumentasi:
http://127.0.0.1:8000/docs
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Custom Layer
@tf.keras.utils.register_keras_serializable(package="WorkSafeAI", name="AttentionPooling")
class AttentionPooling(tf.keras.layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.score_dense = tf.keras.layers.Dense(units, activation="tanh")
        self.attention_score = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        scores = self.attention_score(self.score_dense(inputs))
        weights = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(inputs * weights, axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


# Path Model dan Artifact
BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "models/nlp_model"

MODEL_PATH = EXPORT_DIR / "worksafe_risk_model_best.keras"
FALLBACK_MODEL_PATH = EXPORT_DIR / "worksafe_risk_model.keras"
ARTIFACT_PATH = EXPORT_DIR / "worksafe_artifacts.json"


# Global Object
app = FastAPI(
    title="WorkSafe AI Risk Prediction API",
    description="API untuk prediksi risiko otomasi pekerjaan dan rekomendasi reskilling menggunakan OpenRouter.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ubah ke domain frontend kamu.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
ARTIFACTS = None


# Schema Request dan Response
class PredictRequest(BaseModel):
    title: str = Field(..., example="Warehouse Administration Staff")
    description: str = Field(
        default="",
        example="Responsible for data entry, inventory documentation, warehouse reports, and repetitive administrative tasks.",
    )
    top_skills: str = Field(
        default="",
        example="data entry, inventory management, spreadsheet, reporting",
    )
    top_core_tasks: str = Field(
        default="",
        example="record inventory movement, prepare reports, check warehouse documents",
    )
    numeric_features: Optional[Dict[str, float]] = Field(
        default=None,
        example={},
        description="Opsional. Jika kosong, akan memakai median dari training artifact.",
    )
    generate_reskilling: bool = Field(
        default=True,
        description="Jika true, API akan memanggil OpenRouter untuk membuat rekomendasi reskilling.",
    )
    openrouter_model: str = Field(
        default="openrouter/free",
        description="Model OpenRouter. Bisa diganti sesuai model yang tersedia di akunmu.",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    artifacts_loaded: bool
    model_path: str
    artifact_path: str


# Load Artifact dan Model
def load_artifacts() -> Dict[str, Any]:
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"File artifact tidak ditemukan: {ARTIFACT_PATH}. "
            "Pastikan notebook training sudah dijalankan sampai cell export artifact."
        )

    with open(ARTIFACT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    selected_model_path = MODEL_PATH

    if not selected_model_path.exists() and FALLBACK_MODEL_PATH.exists():
        selected_model_path = FALLBACK_MODEL_PATH

    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan. Dicari di: {MODEL_PATH} atau {FALLBACK_MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        selected_model_path,
        custom_objects={
            "AttentionPooling": AttentionPooling,
            "WorkSafeAI>AttentionPooling": AttentionPooling,
        },
        compile=False,
    )

    return model


@app.on_event("startup")
def startup_event():
    global MODEL, ARTIFACTS

    print("Load artifacts...")
    ARTIFACTS = load_artifacts()

    print("Load model...")
    MODEL = load_model()

    print("Model dan artifact berhasil diload.")
    print("Input model:", [inp.name for inp in MODEL.inputs])
    print("Output model:", [out.name for out in MODEL.outputs])


# Helper Input
def build_job_text(title: str, description: str = "", top_skills: str = "", top_core_tasks: str = "") -> str:
    return (
        f"Job Title: {title}. "
        f"Description: {description}. "
        f"Top Skills: {top_skills}. "
        f"Core Tasks: {top_core_tasks}."
    )


def build_numeric_array(artifacts: Dict[str, Any], numeric_features: Optional[Dict[str, float]] = None) -> np.ndarray:
    numeric_features = numeric_features or {}

    numeric_cols = artifacts.get("numeric_cols", [])
    numeric_medians = artifacts.get("numeric_medians", {})

    values = []
    for col in numeric_cols:
        value = numeric_features.get(col, numeric_medians.get(col, 0.0))
        values.append(float(value))

    return np.array([values], dtype="float32")


# Helper JSON AI
def parse_ai_json(text: str) -> Dict[str, Any]:
    if not text:
        return {
            "error": "AI response kosong.",
            "raw_response": text,
        }

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "error": "Response AI bukan JSON valid.",
        "raw_response": text,
    }


# OpenRouter Reskilling Generator
def generate_reskilling_with_openrouter(
    title: str,
    description: str,
    top_skills: str,
    top_core_tasks: str,
    risk_label: str,
    confidence: float,
    automation_risk_score: float,
    model_name: str = "meta-llama/llama-3.3-70b-instruct:free",
) -> Dict[str, Any]:
    api_key = ""

    if not api_key:
        return {
            "openrouter_status": "failed",
            "error_type": "MissingAPIKey",
            "message": "OPENROUTER_API_KEY belum di-set di environment.",
        }

    try:
        from openrouter import OpenRouter
    except ImportError:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "error_type": "ImportError",
            "message": "Package openrouter belum terinstall. Install dengan: pip install openrouter",
        }

    risk_percent = round(float(automation_risk_score) * 100, 2)

    input_payload = {
        "job_input": {
            "title": title,
            "description": description,
            "top_skills": top_skills,
            "top_core_tasks": top_core_tasks,
        },
        "model_prediction": {
            "risk_label": risk_label,
            "confidence": round(float(confidence), 4),
            "automation_risk_score": round(float(automation_risk_score), 4),
            "risk_percent": risk_percent,
        },
    }

    system_prompt = """
Kamu adalah AI career coach untuk aplikasi WorkSafe AI.
Tugasmu membuat rekomendasi reskilling pekerjaan berdasarkan input pekerjaan user dan hasil prediksi model NLP.

Balas HANYA dalam format JSON valid.
Jangan memakai markdown.
Jangan menambahkan penjelasan di luar JSON.
Gunakan bahasa Indonesia yang jelas, realistis, dan praktis untuk pekerja Indonesia.
Rekomendasi harus dibuat dinamis dari konteks input user.
""".strip()

    user_prompt = f"""
Buat rekomendasi reskilling dinamis berdasarkan data berikut:

{json.dumps(input_payload, indent=2, ensure_ascii=False)}

Format JSON yang wajib dikembalikan:

{{
  "recommendation_source": "openrouter_gen_ai",
  "risk_interpretation": "penjelasan singkat tentang arti level risiko pekerjaan ini",
  "main_reskilling_goal": "tujuan utama reskilling untuk user",
  "recommended_skills": [
    {{
      "skill": "nama skill",
      "reason": "alasan skill ini penting",
      "priority": "High/Medium/Low"
    }}
  ],
  "learning_roadmap": [
    {{
      "phase": "Minggu 1-2",
      "focus": "fokus belajar",
      "activities": ["aktivitas 1", "aktivitas 2"]
    }}
  ],
  "tools_to_learn": ["tool 1", "tool 2"],
  "mini_project_ideas": ["ide project 1", "ide project 2"],
  "career_transition_options": ["opsi karier 1", "opsi karier 2"],
  "estimated_learning_duration": "estimasi durasi belajar"
}}

Aturan:
- recommended_skills minimal 5 item.
- learning_roadmap minimal 3 fase.
- tools_to_learn minimal 4 item.
- mini_project_ideas minimal 3 item.
- career_transition_options minimal 3 item.
- Sesuaikan rekomendasi dengan risk_label dan automation_risk_score.
- Jangan memberi rekomendasi yang terlalu umum.
- Jawab hanya JSON valid.
""".strip()

    try:
        with OpenRouter(api_key=api_key) as client:
            response = client.chat.send(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        message = response.choices[0].message
        content = message.content if hasattr(message, "content") else message["content"]

        parsed = parse_ai_json(content)

        if isinstance(parsed, dict):
            parsed.setdefault("recommendation_source", "openrouter_gen_ai")
            parsed.setdefault("openrouter_status", "success")
            parsed.setdefault("openrouter_model", model_name)

        return parsed

    except Exception as e:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "openrouter_model": model_name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "message": (
                "Prediksi risiko berhasil, tetapi rekomendasi reskilling gagal dibuat "
                "karena request OpenRouter gagal. Coba ulang beberapa menit lagi, "
                "gunakan model lain, atau pastikan akun OpenRouter memiliki limit/credit."
            ),
        }


# Core Prediction
def predict_risk_core(payload: PredictRequest) -> Dict[str, Any]:
    if MODEL is None or ARTIFACTS is None:
        return {
            "status": "error",
            "message": "Model atau artifact belum berhasil diload.",
        }

    label_classes = ARTIFACTS.get("label_classes", [])

    job_text = build_job_text(
        title=payload.title,
        description=payload.description,
        top_skills=payload.top_skills,
        top_core_tasks=payload.top_core_tasks,
    )

    text_arr = np.array([job_text], dtype=object)
    numeric_arr = build_numeric_array(ARTIFACTS, payload.numeric_features)

    try:
        pred_label, pred_score = MODEL.predict(
            {
                "job_text": text_arr,
                "numeric_features": numeric_arr,
            },
            verbose=0,
        )
    except Exception:
        pred_label, pred_score = MODEL.predict(
            [text_arr, numeric_arr],
            verbose=0,
        )

    label_idx = int(np.argmax(pred_label[0]))
    risk_label = label_classes[label_idx] if label_classes else str(label_idx)
    confidence = float(pred_label[0][label_idx])
    risk_score = float(pred_score[0][0])

    result = {
        "status": "success",
        "input": {
            "title": payload.title,
            "description": payload.description,
            "top_skills": payload.top_skills,
            "top_core_tasks": payload.top_core_tasks,
        },
        "prediction": {
            "risk_label": risk_label,
            "confidence": round(confidence, 4),
            "automation_risk_score": round(risk_score, 4),
            "risk_percent": round(risk_score * 100, 2),
        },
    }

    if payload.generate_reskilling:
        result["reskilling_recommendation"] = generate_reskilling_with_openrouter(
            title=payload.title,
            description=payload.description,
            top_skills=payload.top_skills,
            top_core_tasks=payload.top_core_tasks,
            risk_label=risk_label,
            confidence=confidence,
            automation_risk_score=risk_score,
            model_name=payload.openrouter_model,
        )

    return result


# Endpoint
@app.get("/")
def root():
    return {
        "message": "WorkSafe AI Risk Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "artifacts_loaded": ARTIFACTS is not None,
        "model_path": str(MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH),
        "artifact_path": str(ARTIFACT_PATH),
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    return predict_risk_core(payload)


# Opsional: jalan langsung pakai python
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api_fastapi_openrouter:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
