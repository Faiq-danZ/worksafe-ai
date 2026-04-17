WorkSafe AI – Deteksi Risiko PHK & Reskilling Otomatis
Deskripsi Proyek

WorkSafe AI adalah aplikasi berbasis Artificial Intelligence yang dirancang untuk mendeteksi risiko Pemutusan Hubungan Kerja (PHK) serta memberikan rekomendasi reskilling secara otomatis. Sistem ini memanfaatkan data tabular dan data teks untuk menghasilkan prediksi yang akurat dan relevan dalam mendukung kesiapan kerja di masa depan.

Proyek ini dikembangkan sebagai bagian dari program DBS Foundation Coding Camp – AI Engineer (Dicoding) dalam bentuk Capstone Project.

Tujuan Proyek
Mengidentifikasi risiko PHK berdasarkan data pengguna
Memberikan rekomendasi reskilling yang sesuai
Mendukung pengambilan keputusan berbasis data
Meningkatkan kesiapan tenaga kerja menghadapi perubahan ekonomi
Struktur Proyek
worksafe-ai/
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
Teknologi yang Digunakan
Python
TensorFlow (Deep Learning)
NumPy & Pandas
Scikit-learn
Natural Language Processing (NLP)
Tim Pengembang
AI Engineer
Ahmad Faiq Zidane — Tabular Risk Score Model
Sefiand Neeza Efendy — NLP Classification Model
Pembagian Tugas AI Engineer
Tabular Model (Risk Score)
Mengembangkan model prediksi risiko PHK berbasis data tabular
Menggunakan TensorFlow Functional API
Mengimplementasikan Custom Loss Function
Melakukan training dan evaluasi model
NLP Model (Text Classification)
Mengembangkan model klasifikasi teks
Menggunakan Custom Layer dan Custom Callback
Melakukan preprocessing teks
Melakukan training dan evaluasi model
Alur Pengembangan Model
Data dikumpulkan dan diproses oleh tim Data Science
Data dibersihkan dan dipersiapkan untuk modeling
Model dikembangkan dan dilatih menggunakan TensorFlow
Model dievaluasi menggunakan metrik yang sesuai
Model disimpan dalam format .keras atau SavedModel
Model digunakan untuk inference dalam aplikasi
Cara Menjalankan Project
1. Clone Repository
git clone https://github.com/Faiq-danZ/worksafe-ai.git
cd worksafe-ai
2. Install Dependencies
pip install -r requirements.txt
3. Training Model
python training/train_tabular.py
4. Inference
python inference/predict_tabular.py
Status Proyek

Tahap pengembangan awal (Checkpoint 1 – Project Planning)

Catatan
Dataset akan disediakan oleh tim Data Science
Disarankan menggunakan Google Colab untuk proses training model
Folder environment seperti venv/ tidak disertakan dalam repository
Lisensi

Proyek ini menggunakan lisensi MIT License.

MIT License

Copyright (c) 2026 WorkSafe AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
