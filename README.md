# ML-projects- 1
# 📧 Email Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A supervised machine learning project that classifies emails as **Spam** or **Not Spam (Ham)** using NLP techniques and popular ML algorithms.

This project demonstrates the application of text preprocessing, feature extraction with TF-IDF, and model training using classifiers like Naive Bayes, Logistic Regression, and more.

---

## 🧠 Features

- ✅ Spam/Ham binary classification
- ✨ Cleaned and preprocessed email text
- 🔤 TF-IDF Vectorization for feature extraction
- 🤖 Multiple ML algorithms (Naive Bayes, Logistic Regression, etc.)
- 📊 Evaluation metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

---

## 📁 Dataset

Uses the dataset provided in the Repo or **Kaggle Email Spam Dataset**, which includes:
- 5,000+ labeled emails
- Balanced distribution of spam and ham emails
- Plain-text format

---

## 🏗️ Tech Stack

| Component         | Tool / Library        |
|------------------|------------------------|
| Programming Lang | Python 3.8+            |
| ML Libraries     | scikit-learn, pandas, NumPy |
| NLP              | NLTK / spaCy (optional) |
| Visualization    | matplotlib, seaborn    |
| Vectorization    | TF-IDF from `sklearn`  |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/faizankd1/email-spam-detector.git
cd email-spam-detector



# ML project-2
# 🎭 Tiny Shakespeare Text Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A lightweight text generation model trained on the Tiny Shakespeare dataset (~1MB of Shakespeare's plays). It generates poetic, dramatic, and stylized text in the flavor of the Bard himself — perfect for creative projects, AI writing experiments, or just having fun with Renaissance English.

---

## ✨ Features

- 🧠 Character-level text generation
- 🕰️ Learned from real Shakespearean dialogue
- 💬 Generate scenes, monologues, or quotes
- ⚡ Fast training and inference on low-end machines

---

## 📂 Dataset

We use the [Tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt), which contains:

- Dialogue from various plays
- Stage directions and scene formatting
- Authentic Shakespearean language (~1 million characters)

---

## 🏗️ Model Details

| Parameter       | Value           |
|----------------|------------------|
| Model Type      | LSTM / GRU / Transformer *(edit as needed)* |
| Training Type   | Character-level |
| Input Length    | 100 characters |
| Framework       | PyTorch *(or TensorFlow if applicable)* |
| Output          | Shakespearean-style text |

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/faizankd1/tiny-shakespeare-generator.git
cd tiny-shakespeare-generator
