# 🏦Fraud-Detection-in-Financial-Transactions
---
Financial fraud is a growing threat in the era of digital banking.
Traditional detection methods often fail to catch sophisticated fraud
patterns in real time. This project uses machine learning to detect
fraudulent bank account opening applications.

---

## Objectives
- Perform EDA to find fraud patterns and anomalies
- Handle hidden missing values (-1 sentinel) and class imbalance
- Study relationship between demographics, device info, and fraud
- Build a fraud detection model using Random Forest
- Evaluate using Precision, Recall, F1-Score and ROC-AUC

---

## Data Source Acknowledgement
**Bank Account Fraud Dataset Suite — NeurIPS 2022**
Developed by Feedzai Research
🔗 [Kaggle Link](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)

- 1 million rows, 32 features
- Target: fraud_bool (0 = Genuine, 1 = Fraud)
- Fraud rate: ~1% — severely imbalanced

---

## Dataset Description
The **Bank Account Fraud (BAF) Dataset Suite** is a synthetic yet
realistic dataset generated from a real-world bank account opening
fraud detection system using state-of-the-art tabular data generation
techniques (CTGAN), with differential privacy applied to protect
customer confidentiality.
| Property | Details |
|----------|---------|
| Total Rows | 1,000,000 |
| Total Columns | 32 |
| Target Variable | fraud_bool |
| Fraud Rate | ~1.18% |
| Time Period | 8 months (Month 0 to Month 7) |

### Target Variable
- **0** — Genuine / Legitimate Application
- **1** — Fraudulent Application
  
### Important Features:

- Credit_risk_score
- Proposed_credit_limit
- Customer_age
- Income
- Employment_status
- Housing_status
- Behavioural indicators

---
### Exploratory Data Analysis (EDA) 
· Target Distribution · Fraud vs Non-Fraud · Feature Distributions · Correlation Analysis · Pair Plot · Missing Value Analysis

| | |
|---------|----------|
| 98.82% legitimate vs 1.18% fraud — severe imbalance | class_weight='balanced'|
| Higher fraud in unemployed, free email users, foreign requests | Used as key features in model |
| Fraud rate rises month over month (temporal drift) | Noted as deployment consideration |
| -1 sentinel values in 6 columns (not missing — not applicable) | Kept as-is + binary flag columns created |
| Mixed numerical, binary and categorical feature types | Separate preprocessing per type |
| Correlation studied across all feature-target combinations | Point-Biserial · Cramér's V · Pearson Heatmap |

---
## ML Model- Random Forest Classifier
**Why Random Forest?**

 - Handles imbalance, negatives(no mathematical issues) and mixed features
 - Large data + gives feature importance 
 - Robust to outliers

**Why NOT other models?**
|Model | Reason|
|-------|----------------|
| Logistic Regression | Assumes linearity — fraud patterns are non-linear |
| KNN | Too slow on 1 million rows |
| SVM | Does not scale to large datasets |
| Naive Bayes | Assumes feature independence — violated here |
| Single Decision Tree | Overfits easily |

---
## Evaluation Metrics

| Metric | Why Used |
|--------|---------|
| Precision | How many flagged frauds are actually fraud? |
| Recall | How many actual frauds did we catch? ← Most important |
| F1-Score | Balance between Precision and Recall |
| ROC-AUC | Overall model discrimination — not affected by imbalance |

> Accuracy is NOT used — a model predicting all legitimate
> still gets 98.8% accuracy while catching zero fraud.

---
## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Google Colab

---

## 📄 Citation
Jesus et al. "Turning the Tables: Biased, Imbalanced, Dynamic
Tabular Datasets for ML Evaluation." NeurIPS 2022.
