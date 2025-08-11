# 🗣️ Sentiment Analyzer API

## 📌 Overview

A **Machine Learning API** for real-time **sentiment analysis** using **scikit-learn** and **FastAPI**.
This project demonstrates core NLP techniques, robust engineering practices, and automated CI/CD with GitHub Actions.

---

## 🎯 Objectives

- Train a text classification model to predict sentiment (positive, negative, neutral).
- Automate linting, testing, and deployment with GitHub Actions.

---

## 📊 Dataset

Uses the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) or any dataset of labeled text.

Example features:
- `review_text`
- `label` (positive, negative)

---

## 🧠 ML Concepts Implemented

### 1️⃣ Text Vectorization
- Converts raw text to numeric features using `CountVectorizer` or `TfidfVectorizer`.

### 2️⃣ Model Training
- Uses Logistic Regression or Support Vector Machine (SVM) for classification.

### 3️⃣ Evaluation
- Train/test split, accuracy, and classification report.
- Save trained model and vectorizer with `joblib`.

---

## ⚙️ DevOps Integration


### 🚀 GitHub Actions CI/CD
Automates:
- Linting (`flake8`)
- Unit tests for model & API
- Coverage reporting

---

## 📚 Deliverables

- ✅ Clean Python scripts for training and serving.
- ✅ Jupyter Notebook for exploration and training.
- ✅ CI/CD workflow with linting & tests.
- ✅ README with full instructions.

---

## 🚀 Usage

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
