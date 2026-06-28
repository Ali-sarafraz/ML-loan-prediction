# 🏦 Loan Eligibility Prediction — ML Classification Pipeline

Automatically predict whether a loan applicant should be **approved or rejected** using three machine learning classifiers, with end-to-end preprocessing, hyperparameter tuning, and model comparison.

---

## 📌 Overview

Banks process thousands of loan applications every day. Manual review is slow and inconsistent. This project trains and compares ML classifiers on historical applicant data to automate the eligibility decision.

**Task:** Binary classification — Approved (`1`) vs. Rejected (`0`)  
**Dataset:** [Analytics Vidhya — Loan Prediction III](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/)

---

## 📊 Results

### Test Set Performance

| Model               | Accuracy | Precision | Recall | F1 Score | AUC   |
|---------------------|:--------:|:---------:|:------:|:--------:|:-----:|
| Logistic Regression | 0.854    | 0.832     | 0.988  | 0.903    | 0.836 |
| K-Nearest Neighbors | 0.740    | 0.743     | 0.953  | 0.835    | 0.763 |
| MLP Neural Network  | **0.870**| 0.848     | 0.988  | **0.913**| 0.836 |

### 5-Fold Cross-Validation

| Model               | Mean Accuracy | Std Dev |
|---------------------|:-------------:|:-------:|
| **Logistic Regression** | **0.805** | ±0.025 |
| MLP Neural Network  | 0.689         | ±0.003  |
| K-Nearest Neighbors | 0.637         | ±0.029  |

### ✅ Best Model: Logistic Regression

While MLP achieves the highest single-split accuracy, Logistic Regression is the recommended model due to its **superior cross-validation score**, stable generalisation, and interpretability. MLP's low CV score (0.689 vs 0.870 test accuracy) signals overfitting to the train/test split.

---

## 🗂️ Project Structure

```
ML-loan-prediction/
│
├── data/
│   ├── loan.csv          # Training data (614 applicants)
│   └── test.csv          # Hold-out inference set (367 applicants)
│
├── Modeling.ipynb        # Full ML pipeline (see below)
└── README.md
```

---

## ⚙️ Pipeline

```
Raw Data
   │
   ├─ 1. Stratified Imputation   (mode for binary, median for numeric, domain rules)
   ├─ 2. Feature Encoding        (LabelEncoder + One-Hot for Property_Area)
   ├─ 3. Train/Test Split        (80/20, stratified)
   ├─ 4. Normalisation           (StandardScaler — fit on train only)
   ├─ 5. GridSearchCV Tuning     (5-fold CV per model)
   ├─ 6. Model Evaluation        (accuracy, precision, recall, F1, AUC)
   └─ 7. Inference               (predictions on hold-out test set)
```

---

## 🧠 Models & Hyperparameter Search

### Logistic Regression
```python
param_grid = {
    "C":       [0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
}
# Best: C=0.1, penalty='l1'
```

### K-Nearest Neighbors
```python
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"],
}
# Best: k=7, weights='uniform', metric='manhattan'
```

### MLP Neural Network
```python
param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "activation":         ["relu", "tanh"],
    "alpha":              [0.0001, 0.001, 0.01],
    "solver":             ["adam", "sgd"],
    ...
}
# Best: (50,50) relu, alpha=0.0001, sgd, lr=0.01
```

---

## 📋 Dataset Features

| Feature            | Type        | Description                            |
|--------------------|-------------|----------------------------------------|
| `Gender`           | Binary      | Male / Female                          |
| `Married`          | Binary      | Applicant marital status               |
| `Dependents`       | Numeric     | Number of dependents (0–3+)            |
| `Education`        | Binary      | Graduate / Not Graduate                |
| `Self_Employed`    | Binary      | Self-employed or not                   |
| `ApplicantIncome`  | Numeric     | Monthly income of applicant            |
| `CoapplicantIncome`| Numeric     | Monthly income of co-applicant         |
| `LoanAmount`       | Numeric     | Loan amount requested (in thousands)   |
| `Loan_Amount_Term` | Numeric     | Term of the loan (in months)           |
| `Credit_History`   | Binary      | Meets credit guidelines (1) or not (0) |
| `Property_Area`    | Categorical | Urban / Semiurban / Rural              |
| `Loan_Status`      | **Target**  | Approved (Y) / Rejected (N)            |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run

```bash
git clone https://github.com/Ali-sarafraz/ML-loan-prediction.git
cd ML-loan-prediction
jupyter notebook Modeling.ipynb
```

---

## 🔍 Key Design Decisions

**Stratified imputation** — Missing values are imputed separately for approved and rejected subsets to preserve class-specific distributions rather than blending them.

**Scaler fitted on train only** — `StandardScaler` is fit exclusively on training data to prevent data leakage into the test set.

**MLP inside a Pipeline** — During grid search, the scaler is part of the CV pipeline so each fold is normalised independently.

**LR without normalisation comparison** — Unlike distance-based models, Logistic Regression is somewhat scale-invariant. Both variants are benchmarked to confirm this empirically.

---

## 🛣️ Future Work

- [ ] EDA notebook with distribution plots and correlation heatmap
- [ ] SHAP values for Logistic Regression interpretability
- [ ] Handle class imbalance with SMOTE or `class_weight='balanced'`
- [ ] Add Random Forest / XGBoost for ensemble comparison
- [ ] Deploy best model as a REST API (FastAPI + Docker)

---

## 📄 License

MIT License — feel free to use and adapt.
