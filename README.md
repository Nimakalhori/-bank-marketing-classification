# 📊 Bank Marketing Classification Project

This repository contains a professional classification project developed as part of a Master's-level Authentication course and refined with real-world data analysis practices. The goal is to predict whether a client will subscribe to a term deposit based on various banking and personal features.

---

## 🔍 Dataset

* **Source**: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
* **Records**: 41,188
* **Features**: 20 original (63 after encoding)
* **Target**: `y` (binary classification - "yes"/"no")

---

## 🧪 Project Structure

```bash
├── data/                             # Raw and cleaned dataset files (optional in .gitignore)
├── notebooks/                        # Jupyter Notebooks for each stage
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_model_training_and_evaluation.ipynb
│   ├── 03_feature_selection_and_tuning.ipynb
│   ├── 04_refinement_classification_methods.ipynb
│   └── 05_complexity_and_execution_time.ipynb
├── optimized_threshold_model.pkl    # Final saved model with threshold
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview (you are here)
```

---

## ✅ Key Highlights

* 🧹 **EDA & Preprocessing**: Cleaned data, encoded categorical features, scaled numerics.
* 🤖 **Model Training**: 9 classifiers tested including Logistic Regression, LDA, SVM, Tree-based.
* 🏆 **Best Model**: Tuned Gradient Boosting with threshold optimization.
* 📈 **Evaluation**: Confusion matrix, F1-score, and McNemar’s test.
* ⚙️ **Refinements**: Threshold optimization, feature interaction, stacking.
* ⏱️ **Execution Time Analysis**: Compared inference time, and performance.

---

## 🧠 Best Results (Optimized Gradient Boosting)

```
Accuracy:        0.92
Precision (class 1): 0.63
Recall (class 1):    0.73
F1-score (class 1):  0.67
```

Reduced feature count: **8 → from 63**
Training time improved and model file size reduced.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Nimakalhori/-bank-marketing-classification.git
cd bank-marketing-classification
```

### 2. Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Use JupyterLab or VS Code to open and run notebooks in sequence:

```bash
jupyter notebook
```

---

## 📦 Requirements

See `requirements.txt` for full environment. Typical setup includes:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib
* statsmodels

---

## 📚 References

* UCI Machine Learning Repository
* scikit-learn documentation
* McNemar statistical test methodology

---

## 📬 Author

**Nima Kalhori**
M.Sc. Student in Information Technology – Specialization: Authentication
Data Analyst at Phoenix Contact

---

## 🏁 License

This project is for educational and academic showcase purposes.
