# WorkSafe AI - AI Module 🤖

WorkSafe AI is a capstone project focused on helping individuals prepare for the future of work by predicting layoff risks and providing reskilling recommendations.

This repository contains the Artificial Intelligence (AI) components of the system, including deep learning models for tabular data and natural language processing (NLP).

---

## 🎯 Project Objective

To build AI models that can:

* Predict the risk of layoffs (PHK) based on employee data
* Provide intelligent recommendations for reskilling using NLP

---

## 🧠 AI Components

### 1. Tabular Risk Prediction Model

* Type: Binary Classification
* Framework: TensorFlow (Functional API)
* Input: Structured/tabular employee data
* Output: Probability of layoff risk

### 2. NLP Classification Model

* Type: Text Classification
* Framework: TensorFlow
* Input: Text data (e.g., job descriptions, skills)
* Output: Reskilling or job-related predictions

---

## 📁 Project Structure

```
worksafe-ai/
│── data/                 # Dataset (final from Data Science team)
│── models/               # Saved models (.keras)
│   ├── tabular_model/
│   └── nlp_model/
│
│── training/             # Training scripts
│   ├── train_tabular.py
│   └── train_nlp.py
│
│── inference/            # Prediction scripts
│   ├── predict_tabular.py
│   └── predict_nlp.py
│
│── utils/                # Helper functions (preprocessing, etc.)
│
│── requirements.txt
│── README.md
```

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn

---

## 🚀 How to Run (Basic)

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Train model

```
python training/train_tabular.py
```

### 3. Run inference

```
python inference/predict_tabular.py
```

---

## 🧪 Model Development Notes

* The tabular model uses **TensorFlow Functional API**
* Includes a **custom loss function** to handle imbalanced data
* Models are saved in `.keras` format for production use

---

## 👥 Team Members

* **Ahmad Faiq Zidane** – Tabular Risk Model (AI Engineer)
* **Sefiand Neeza Efendy** – NLP Model (AI Engineer)

---

## 📌 Notes

* Dataset is provided and processed by the Data Science team
* This repository focuses only on AI model development
* Further integration will be handled with the backend team

---

## 📄 License

This project is developed for educational purposes as part of Dicoding DBS Foundation Capstone Program.
