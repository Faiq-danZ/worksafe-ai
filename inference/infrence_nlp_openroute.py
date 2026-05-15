"""
Struktur folder:
project/
├── inference_nlp_openroute.py
└── models/nlp_model/
    ├── worksafe_risk_model_best.keras
    └── worksafe_artifacts.json

Langkah pakai:
python -m venv venv, jika sudah ada aktifkan venv
Aktifkan:
Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

pip install tensorflow numpy openrouter
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import tensorflow as tf


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
# Artifact hanya dipakai untuk:
# - label_classes
# - numeric_cols
# - numeric_medians

BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "models/nlp_model"

MODEL_PATH = EXPORT_DIR / "worksafe_risk_model_best.keras"
FALLBACK_MODEL_PATH = EXPORT_DIR / "worksafe_risk_model.keras"
ARTIFACT_PATH = EXPORT_DIR / "worksafe_artifacts.json"


def load_artifacts():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"File artifact tidak ditemukan: {ARTIFACT_PATH}\n"
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
            f"File model tidak ditemukan.\n"
            f"Dicari di:\n"
            f"- {MODEL_PATH}\n"
            f"- {FALLBACK_MODEL_PATH}\n"
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


def build_job_text(title, description="", top_skills="", top_core_tasks=""):
    return (
        f"Job Title: {title}. "
        f"Description: {description}. "
        f"Top Skills: {top_skills}. "
        f"Core Tasks: {top_core_tasks}."
    )


def build_numeric_array(artifacts, numeric_features=None):
    numeric_features = numeric_features or {}

    numeric_cols = artifacts.get("numeric_cols", [])
    numeric_medians = artifacts.get("numeric_medians", {})

    values = []
    for col in numeric_cols:
        value = numeric_features.get(col, numeric_medians.get(col, 0.0))
        values.append(float(value))

    return np.array([values], dtype="float32")


def parse_ai_json(text):
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


def generate_reskilling_with_openrouter(
    title,
    description,
    top_skills,
    top_core_tasks,
    risk_label,
    confidence,
    automation_risk_score,
    model_name="meta-llama/llama-3.3-70b-instruct:free",
):
    """
    Rekomendasi reskilling dibuat dari OpenRouter.
    """
    api_key = ""

    if not api_key:
        return {
            "recommendation_source": "openrouter_gen_ai",
            "openrouter_status": "failed",
            "error_type": "MissingAPIKey",
            "message": (
                "OPENROUTER_API_KEY belum di-set. "
                "Mac/Linux: export OPENROUTER_API_KEY='isi_api_key_kamu'. "
                "Windows PowerShell: $env:OPENROUTER_API_KEY='isi_api_key_kamu'."
            ),
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
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
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


def predict_risk(
    model,
    artifacts,
    title,
    description="",
    top_skills="",
    top_core_tasks="",
    numeric_features=None,
    openrouter_model="openrouter/free",
    generate_reskilling=True,
):
    label_classes = artifacts.get("label_classes", [])

    job_text = build_job_text(
        title=title,
        description=description,
        top_skills=top_skills,
        top_core_tasks=top_core_tasks,
    )

    text_arr = np.array([job_text], dtype=object)
    numeric_arr = build_numeric_array(artifacts, numeric_features)

    try:
        pred_label, pred_score = model.predict(
            {
                "job_text": text_arr,
                "numeric_features": numeric_arr,
            },
            verbose=0,
        )
    except Exception:
        pred_label, pred_score = model.predict(
            [text_arr, numeric_arr],
            verbose=0,
        )

    label_idx = int(np.argmax(pred_label[0]))
    risk_label = label_classes[label_idx] if label_classes else str(label_idx)
    confidence = float(pred_label[0][label_idx])
    risk_score = float(pred_score[0][0])

    result = {
        "title": title,
        "risk_label": risk_label,
        "confidence": round(confidence, 4),
        "automation_risk_score": round(risk_score, 4),
        "risk_percent": round(risk_score * 100, 2),
    }

    if generate_reskilling:
        result["reskilling_recommendation"] = generate_reskilling_with_openrouter(
            title=title,
            description=description,
            top_skills=top_skills,
            top_core_tasks=top_core_tasks,
            risk_label=risk_label,
            confidence=confidence,
            automation_risk_score=risk_score,
            model_name=openrouter_model,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Load model WorkSafe AI NLP, prediksi risiko, dan generate reskilling dengan OpenRouter."
    )
    parser.add_argument("--title", default="Warehouse Administration Staff")
    parser.add_argument(
        "--description",
        default="Responsible for data entry, inventory documentation, warehouse reports, and repetitive administrative tasks.",
    )
    parser.add_argument(
        "--top_skills",
        default="data entry, inventory management, spreadsheet, reporting",
    )
    parser.add_argument(
        "--top_core_tasks",
        default="record inventory movement, prepare reports, check warehouse documents",
    )
    parser.add_argument(
        "--numeric_features",
        default="{}",
        help='Opsional. Format JSON string, contoh: \'{"feature_a": 1.0, "feature_b": 2.0}\'',
    )
    parser.add_argument(
        "--openrouter_model",
        default="openrouter/free",
        help='Default: "openrouter/free". Bisa diganti model ID OpenRouter lain.',
    )
    parser.add_argument(
        "--no_reskilling",
        action="store_true",
        help="Gunakan ini jika hanya ingin test prediksi model tanpa OpenRouter.",
    )

    args = parser.parse_args()

    try:
        numeric_features = json.loads(args.numeric_features)
    except json.JSONDecodeError as e:
        raise ValueError(
            "--numeric_features harus berupa JSON valid. Contoh: "
            '\'{"employment_count": 1000, "median_wage": 50000}\''
        ) from e

    print("Load artifacts...")
    artifacts = load_artifacts()

    print("Load model...")
    model = load_model()

    print("Model berhasil diload.")
    print("Input model:", [inp.name for inp in model.inputs])
    print("Output model:", [out.name for out in model.outputs])

    result = predict_risk(
        model=model,
        artifacts=artifacts,
        title=args.title,
        description=args.description,
        top_skills=args.top_skills,
        top_core_tasks=args.top_core_tasks,
        numeric_features=numeric_features,
        openrouter_model=args.openrouter_model,
        generate_reskilling=not args.no_reskilling,
    )

    print("\nHasil prediksi:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
