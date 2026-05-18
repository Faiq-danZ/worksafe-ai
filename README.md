<div align="center">

<img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
<img src="https://img.shields.io/badge/Status-Checkpoint%202%20✓-2ECC71?style=for-the-badge"/>

<br/><br/>

```
 ██╗    ██╗ ██████╗ ██████╗ ██╗  ██╗███████╗ █████╗ ███████╗███████╗     █████╗ ██╗
 ██║    ██║██╔═══██╗██╔══██╗██║ ██╔╝██╔════╝██╔══██╗██╔════╝██╔════╝    ██╔══██╗██║
 ██║ █╗ ██║██║   ██║██████╔╝█████╔╝ ███████╗███████║█████╗  █████╗      ███████║██║
 ██║███╗██║██║   ██║██╔══██╗██╔═██╗ ╚════██║██╔══██║██╔══╝  ██╔══╝      ██╔══██║██║
 ╚███╔███╔╝╚██████╔╝██║  ██║██║  ██╗███████║██║  ██║██║     ███████╗    ██║  ██║██║
  ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝    ╚═╝  ╚═╝╚═╝
```

### Deteksi Risiko PHK & Reskilling Otomatis
#### *AI-Powered Workforce Risk Intelligence System*

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Faiq-danZ/worksafe-ai)
&nbsp;
<img src="https://img.shields.io/badge/Accuracy-91.61%25-brightgreen?style=flat-square"/>
&nbsp;
<img src="https://img.shields.io/badge/Dataset-O*NET-blue?style=flat-square"/>
&nbsp;
<img src="https://img.shields.io/badge/DBS%20Foundation-Capstone%20Project-red?style=flat-square"/>

</div>

---

## Tentang Proyek

**WorkSafe AI** adalah sistem kecerdasan buatan yang dirancang untuk mendeteksi tingkat risiko Pemutusan Hubungan Kerja (PHK) pada berbagai jenis pekerjaan berdasarkan profil kompetensi pengguna. Sistem ini juga memberikan rekomendasi reskilling secara otomatis agar tenaga kerja tetap relevan di tengah perkembangan teknologi dan otomasi.

Proyek ini dikembangkan dalam program **DBS Foundation Coding Camp — AI Engineer (Dicoding)** sebagai Capstone Project dengan tema **Future-Ready Work & Economy**.

---

## Tujuan Proyek

- Mengidentifikasi risiko PHK berdasarkan profil skill dan aktivitas kerja pengguna
- Memberikan output prediksi berupa tiga kategori: **Low Risk**, **Medium Risk**, **High Risk**
- Mendukung pengambilan keputusan berbasis data untuk pengembangan karir
- Meningkatkan kesiapan tenaga kerja Indonesia menghadapi perubahan ekonomi dan otomasi

---

## Hasil Model

| Metrik | Nilai |
|---|---|
| Test Accuracy | **91.61%** |
| F1-Score (rata-rata) | **91%** |
| Recall High Risk | **100%** |
| Format Model | `.keras` (TF Production Ready) |
| Syarat Minimum Capstone | **85%** ✓ |

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Deep Learning Framework | TensorFlow 2.x + Keras Functional API |
| Custom Component | Focal Loss (Custom Loss Function) |
| Preprocessing | Scikit-learn (StandardScaler, SimpleImputer) |
| Data Processing | Pandas, NumPy |
| REST API | Flask |
| Dataset | O\*NET (Skills, Knowledge, Work Activities) |
| Environment | Google Colab |

---

## Struktur Folder

```
worksafe-ai/
│
├── data/
│   └── .gitkeep
│
├── models/
│   ├── tabular_model/
│   │   ├── worksafe_model_v1.keras
│   │   ├── scaler.pkl
│   │   ├── imputer.pkl
│   │   └── feature_cols.pkl
│   └── nlp_model/
│       └── worksafe_artifacts.json
│
├── training/
│   ├── preprocessing_tabular.ipynb
│   ├── train_tabular.ipynb
│   └── predict_tabular.ipynb
│
├── inference/
│   ├── predict_tabular.py
│   ├── api_tabular.py
│   └── infrence_nlp_openroute.py
│
├── utils/
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/Faiq-danZ/worksafe-ai.git
cd worksafe-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocessing Data

Buka dan jalankan semua cell pada:
```
training/preprocessing_tabular.ipynb
```
Output: `data_siap_training.csv`, `scaler.pkl`, `imputer.pkl`, `feature_cols.pkl`

### 4. Training Model

Buka dan jalankan semua cell pada:
```
training/train_tabular.ipynb
```
Output: `worksafe_model_v1.keras`

### 5. Inference (Notebook)

Buka dan jalankan:
```
training/predict_tabular.ipynb
```

### 6. Jalankan REST API (Flask)

Dijalankan di komputer lokal atau server backend, bukan di Colab:
```bash
python inference/api_tabular.py
```

Server berjalan di `http://localhost:5000`

**Endpoint yang tersedia:**

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/` | Cek status server |
| POST | `/predict` | Prediksi risiko pekerjaan |
| GET | `/features` | Lihat daftar fitur model |

**Contoh request ke `/predict`:**

```json
POST /predict
Content-Type: application/json

{
  "features": {
    "skl_critical_thinking": 4.5,
    "act_working_with_computers": 5.0,
    "skl_active_learning": 4.0
  }
}
```

**Contoh response:**

```json
{
  "risk_label": "Low Risk",
  "risk_class": 0,
  "confidence": 92.30,
  "probabilities": {
    "Low Risk": 92.30,
    "Medium Risk": 6.10,
    "High Risk": 1.60
  }
}
```

---

## Arsitektur Model Tabular

```
Input (50 fitur)
      │
   Dense(256) → BatchNorm → ReLU → Dropout(0.3)
      │
   Dense(128) → BatchNorm → ReLU → Dropout(0.3)
      │                                  │
   Dense(64)  → BatchNorm → ReLU    Skip Connection
      │                                  │
      └──────────── Add ─────────────────┘
                     │
               ReLU → Dropout(0.15)
                     │
                Dense(32) → ReLU
                     │
              Dense(3) → Softmax
                     │
          [Low Risk | Medium Risk | High Risk]
```

**Custom Loss Function: Focal Loss**
```
FL(p_t) = -α × (1 - p_t)^γ × log(p_t)
γ = 2.0  |  α = 0.25
```

---

## Tim Pengembang

<table>
  <tr>
    <th>Nama</th>
    <th>Divisi</th>
    <th>Tugas</th>
  </tr>
  <tr>
    <td>Ahmad Faiq Zidane</td>
    <td>AI Engineer</td>
    <td>Tabular Risk Score Model (TF Functional API + Custom Loss Function)</td>
  </tr>
  <tr>
    <td>Sefiand Neeza Efendy</td>
    <td>AI Engineer</td>
    <td>NLP Classification Model (Custom Layer + Custom Callback)</td>
  </tr>
  <tr>
    <td colspan="3"><i>+ 2 Web Developer + 2 Data Scientist</i></td>
  </tr>
</table>

---

## Progress Checkpoint

| Checkpoint | Periode | Status | Keterangan |
|---|---|---|---|
| Checkpoint 1 | 1 – 17 April 2026 | ✅ Selesai | Project planning & prototype model |
| Checkpoint 2 | 18 April – 17 Mei 2026 | ✅ Selesai | Training model & evaluasi |
| Checkpoint 3 | 18 Mei – 7 Juni 2026 | 🔄 On Progress | Integrasi & finalisasi |

---

## Catatan Teknis

- Model dilatih menggunakan dataset **O\*NET** (Occupational Information Network) dari Departemen Tenaga Kerja Amerika Serikat
- Label risiko dibuat secara synthetic mengikuti pendekatan **Frey & Osborne (2013)** tentang otomasi pekerjaan
- Proses training menggunakan **Google Colab** karena keterbatasan resource lokal
- File model (`.keras`, `.pkl`) tidak diupload ke GitHub karena ukuran besar, disimpan di Google Drive
- Untuk menjalankan `api_tabular.py`, pastikan 4 file model sudah ada di folder `models/tabular_model/`

---

<div align="center">

**DBS Foundation Coding Camp × Dicoding — AI Engineer**
<br/>
*Capstone Project 2026 | Future-Ready Work & Economy*

</div>
