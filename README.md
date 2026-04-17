# WorkSafe AI - Deteksi Risiko PHK & Reskilling Otomatis

## Deskripsi Proyek

WorkSafe AI adalah aplikasi berbasis Artificial Intelligence yang dirancang untuk mendeteksi risiko Pemutusan Hubungan Kerja (PHK) serta memberikan rekomendasi reskilling secara otomatis. Sistem ini memanfaatkan data tabular dan data teks untuk menghasilkan prediksi yang akurat dan relevan dalam mendukung kesiapan kerja di masa depan.

Proyek ini dikembangkan sebagai bagian dari program DBS Foundation Coding Camp - AI Engineer (Dicoding) dalam bentuk Capstone Project.

## Tujuan Proyek

- Mengidentifikasi risiko PHK berdasarkan data pengguna
- Memberikan rekomendasi reskilling yang sesuai
- Mendukung pengambilan keputusan berbasis data
- Meningkatkan kesiapan tenaga kerja menghadapi perubahan ekonomi

## Teknologi yang Digunakan

- Python
- TensorFlow (Deep Learning)
- NumPy & Pandas
- Scikit-learn
- Natural Language Processing (NLP)

## Tim Pengembang
    |      |
### AI Engineer

- Ahmad Faiq Zidane - Tabular Risk Score Model
- Sefiand Neeza Efendy - NLP Classification Model

## Pembagian Tugas AI Engineer

### Tabular Model (Risk Score)

- Mengembangkan model prediksi risiko PHK berbasis data tabular
- Menggunakan TensorFlow Functional API
- Mengimplementasikan Custom Loss Function
- Melakukan training dan evaluasi model

### NLP Model (Text Classification)

- Mengembangkan model klasifikasi teks
- Menggunakan Custom Layer dan Custom Callback
- Melakukan preprocessing teks
- Melakukan training dan evaluasi model

## Alur Pengembangan Model

1. Data dikumpulkan dan diproses oleh tim Data Science
2. Data dibersihkan dan dipersiapkan untuk modeling
3. Model dikembangkan dan dilatih menggunakan TensorFlow
4. Model dievaluasi menggunakan metrik yang sesuai
5. Model disimpan dalam format .keras atau SavedModel
6. Model digunakan untuk inference dalam aplikasi

## Cara Menjalankan Project

### 1. Clone Repository

```bash
git clone https://github.com/Faiq-danZ/worksafe-ai.git
cd worksafe-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Training Model

```bash
python training/train_tabular.py
```

### 4. Inference

```bash
=======
│
├── data/
│   └── .gitkeep
│
├── models/
│   ├── tabular_model/
│   │   └── .gitkeep
│   └── nlp_model/
│       └── .gitkeep
│
├── training/
│   ├── train_tabular.py
│   └── train_nlp.py
│
├── inference/
│   ├── predict_tabular.py
│   └── predict_nlp.py
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py
│
├── requirements.txt
└── README.md

## Status Proyek

Tahap pengembangan awal (Checkpoint 1 - Project Planning)

## Catatan

- Dataset akan disediakan oleh tim Data Science
- Disarankan menggunakan Google Colab untuk proses training model
- Folder environment seperti venv/ tidak disertakan dalam repository
===================================================================

