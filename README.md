# 💬 Toxic Comment Classification System

> Multi-label toxic comment detection using GloVe embeddings and GRU-based deep learning.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-DeepLearning-green?style=flat-square)
![GloVe](https://img.shields.io/badge/Embeddings-GloVe-purple?style=flat-square)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Features](#2-key-features)
3. [How It Works](#3-how-it-works)
4. [Dataset & Preprocessing](#4-dataset--preprocessing)
5. [Model Architecture](#5-model-architecture)
6. [Tech Stack](#6-tech-stack)
7. [Folder Structure](#7-folder-structure)
8. [How to Run](#8-how-to-run)
9. [Output](#9-output)
10. [Model Comparison & Justification](#10-model-comparison--justification)
11. [Future Improvements](#11-future-improvements)
12. [Conclusion](#12-conclusion)

---

## 1. Overview

This project presents a **multi-label Toxic Comment Classification System** designed to automatically detect harmful content in online platforms.

The system classifies comments into multiple categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

It uses **GloVe embeddings + GRU network** to balance performance, efficiency, and deployability.

---

## 2. Key Features

- 🧠 Multi-label classification (6 toxicity categories)
- 🔤 Advanced text preprocessing pipeline
- 📊 Feature engineering + EDA
- ⚡ Efficient GloVe + GRU architecture
- 🚀 Real-time prediction support
- 💾 Saved model + tokenizer for deployment
- 📈 High validation accuracy (~98–99%)

---

## 3. How It Works

```
Raw Comment
↓
Text Cleaning & Preprocessing
↓
Tokenization (Keras)
↓
Sequence Padding
↓
GloVe Embedding Layer
↓
GRU Layer (Context Learning)
↓
Dense Layers
↓
Sigmoid Output (Multi-label)
↓
Thresholding
↓
Predicted Toxic Labels
```

---

## 4. Dataset & Preprocessing

### 📂 Dataset

- Combined **5 datasets**
- Total size: **~383,000 comments**
- Labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

---

### ⚙️ Preprocessing Pipeline

The system uses a multi-step cleaning pipeline for robust text normalization.

#### Step 1 — Cleaning

- Lowercasing
- HTML & URL removal
- Emoji handling
- Profanity normalization

#### Step 2 — Text Normalization

- Stopword removal
- Lemmatization
- Repeated character reduction

#### Step 3 — Feature Engineering

- Word count
- Caps ratio
- Punctuation features

#### Step 4 — Final Dataset

- Cleaned text stored as `clean_comment`
- Used for model training

---

### 💡 Key Insights

- Severe class imbalance (e.g., threat < 1%)
- Toxic labels often co-occur
- Clean text significantly improves performance

---

### 🧪 Included Assets

- ✅ Preprocessed dataset is already provided
- ✅ Pre-trained model is included — no retraining required

---

## 5. Model Architecture

### Model Type

Convolutional Neural Network with GloVe Embeddings (GloVe + GRU)

### Input

Tokenized & padded text (length = 150)

### Architecture

```
Embedding (GloVe, 100d, frozen)
→ GRU (128 units)
→ Dropout (0.3)
→ Dense (64, ReLU)
→ Dense (6, Sigmoid)
```

---

### ⚙️ Training Strategy

- Loss: Binary Crossentropy
- Optimizer: Adam
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau

---

### 📦 Included Model

Pre-trained model provided: `model.h5` + `tokenizer.pkl`

Enables direct execution without retraining.

---

## 6. Tech Stack

| Technology         | Purpose          |
|--------------------|------------------|
| Python             | Core development |
| TensorFlow / Keras | Model building   |
| GloVe              | Word embeddings  |
| Scikit-learn       | Data splitting   |
| Pandas / NumPy     | Data processing  |

---

## 7. Folder Structure

```
TOXIC-COMMENT-PROJECT/
│
├── processed_dataset*.csv     # Cleaned datasets
├── model.h5                   # Trained model
├── tokenizer.pkl              # Tokenizer
├── app.py                     # Flask backend
├── templates/
│   └── index.html             # Frontend
├── glove.6B.100d.txt          # GloVe embeddings
├── notebook.ipynb             # Training notebook
└── README.md                  # Project documentation
```

---

## 8. How to Run

### Step 1 — Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name
cd your-repo-name
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run the Application

The project already includes the preprocessed dataset and trained model, so no additional preprocessing or training is required.

```bash
python app.py
```

> ⚠️ **Note:** Make sure `glove.6B.100d.txt` is placed in the root directory before running.

---

## 9. Output

The system provides:

| Output              | Description                   |
|---------------------|-------------------------------|
| Toxic labels        | Predicted toxicity categories |
| Probability scores  | Confidence score per label    |
| Real-time detection | Instant prediction via web UI |

### Example

```json
{
  "labels": ["toxic", "obscene", "insult"],
  "scores": {
    "toxic": 0.99,
    "obscene": 0.71,
    "insult": 0.97
  }
}
```

---

## 10. Model Comparison & Justification

### 🔹 Models Tried

| Model               | Result                | Issue              |
|---------------------|-----------------------|--------------------|
| Logistic Regression | Low accuracy          | No context         |
| LSTM                | Moderate              | Slow               |
| DistilBERT          | ~0.60 F1              | Heavy & unstable   |
| DeBERTa             | High accuracy, F1 = 0 | Failed predictions |

---

### ✅ Why GloVe + GRU?

- Efficient & lightweight
- Captures sequential context
- Stable performance
- Works well with cleaned data
- Suitable for real-time deployment

> 👉 Best trade-off between **performance + speed + reliability**

---

## 11. Future Improvements

- [ ] Bi-GRU with Attention mechanism
- [ ] Transformer + GRU hybrid model
- [ ] Better imbalance handling (Focal Loss)
- [ ] Explainability (SHAP / LIME)
- [ ] Multilingual support
- [ ] Cloud deployment (Render / HuggingFace)

---

## 👥 Contributors

| Name    | GitHub |
|---------|--------|
| Ishant  | [@ishant212](https://github.com/ishant212) |
| Lakshya | [@x-lucky-x](https://github.com/X-ImLucky-X) |

---

## 12. Conclusion

This project demonstrates a complete **NLP pipeline** combining:

- Text Preprocessing
- Feature Engineering
- Deep Learning

It serves as a strong portfolio project for roles in **AI/ML Engineering**, **NLP**, and **Content Moderation Systems**.

> 👉 **Well-designed simple models can outperform complex ones when optimized properly.**

The system is **accurate, efficient, and production-ready** for toxic content moderation.

---

> ⭐ If you found this project useful, consider giving it a star on GitHub!
