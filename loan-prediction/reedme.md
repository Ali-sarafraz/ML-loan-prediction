# 🏦 Loan Approval Prediction – Machine Learning Project

This project aims to predict whether a loan will be approved or not based on applicant details using various machine learning models. The dataset used is a binary classification dataset with both categorical and numerical features.

---

## 📁 Project Structure

loan-prediction/
│
├── data/ # Contains original and preprocessed datasets     
├── notebooks/ # Jupyter notebooks for EDA and modeling     
├── README.md # Project overview    
└── requirements.txt # Required libraries    

---

## 📊 Dataset Overview

- **Target Variable**: `Loan_Status`  
    - Y = Approved  
    - N = Not Approved  
- **Features include**:
    - Gender, Married, Dependents, Education, Self_Employed
    - ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
    - Credit_History, Property_Area

---

## ⚙️ Tools & Technologies

- **Language**: Python
- **Libraries**:
    - `pandas`, `numpy`, `matplotlib`, `seaborn`
    - `scikit-learn` for modeling, evaluation, cross-validation

---

## 🔧 Usage

### ▶️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ali-sarafraz/loan-prediction.git
   cd loan-prediction

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt

4. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook

5. **Open and run the notebook inside** `notebooks/Loan_Prediction.ipynb`

Follow the cells in order to:

- Preprocess the dataset

- Train models

- Evaluate and compare performance

- View confusion matrices and plots

---

## 🔬 Methodology

### 1. **Preprocessing**
- Handling missing values using `SimpleImputer`
- Encoding categorical features using `pd.get_dummies`
- Feature scaling with `StandardScaler`

### 2. **Models Trained**
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP Neural Network)

### 3. **Evaluation**
- Metrics: Accuracy, Precision, Recall, F1 Score, AUC
- Confusion Matrix visualization
- **Cross-validation** with `StratifiedKFold` to ensure generalization

---

## 📈 Results Summary

| Model                 | Accuracy (Test Split) | CV Accuracy (Mean ± Std) | F1 Score | AUC     |
|----------------------|------------------------|---------------------------|----------|---------|
| Logistic Regression  | 0.854                  | **0.805 ± 0.025**         | 0.903    | 0.836   |
| K-Nearest Neighbors  | 0.740                  | 0.637 ± 0.029             | 0.835    | 0.763   |
| MLP Neural Network   | **0.870**              | 0.689 ± 0.003             | **0.913**| 0.836   |

---

## ✅ Final Model Recommendation

Although the MLP model showed the highest accuracy on the test split, it had **poor generalization** in cross-validation. Therefore, **Logistic Regression** was selected as the final model due to:

- Strong performance across all metrics  
- Highest cross-validation accuracy  
- Balanced prediction behavior  
- Low variance and good generalization

---

## 🧠 Author

**[Ali Sarafraz]**  
Machine Learning Student | Data Science Enthusiast  
Email: ali.sarafraz530@gmail.com  
GitHub: [Ali-sarafraz](https://github.com/Ali-sarafraz)

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
