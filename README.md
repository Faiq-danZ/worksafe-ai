# worksafe-ai

Project scaffold for tabular and NLP model training/inference.

## Struktur

```text
worksafe-ai/
│── data/
│── models/
│   ├── tabular_model/
│   └── nlp_model/
│── training/
│   ├── train_tabular.py
│   └── train_nlp.py
│── inference/
│   ├── predict_tabular.py
│   └── predict_nlp.py
│── utils/
│── requirements.txt
│── README.md
```

## Quick Start

1. Create virtual env and activate it.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training script example:
   ```bash
   python training/train_tabular.py
   ```
