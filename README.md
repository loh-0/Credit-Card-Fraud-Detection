# Credit Card Fraud Detection Using Decision Trees

This project uses machine learning to detect fraudulent transactions from a large credit card dataset. Fraud is rare (and costly), so we apply **resampling**, **decision trees**, and performance evaluation techniques to build a robust fraud detection pipeline.

---

## Problem Statement

Financial fraud is rare but serious — it affects trust, costs millions, and is hard to detect.  
Out of 250,000+ transactions, **only ~0.17% are fraudulent**. This extreme class imbalance poses a challenge for most classification models.

---

## 📊 Dataset Overview

- **Source**: Publicly available anonymised credit card transaction dataset  
- **Size**: ~250,000 records  
- **Features**: 
  - Numerical transaction details (V1–V28)
  - `Amount` (transaction value)
  - `Time` (seconds elapsed)
  - `Class` (0 = legit, 1 = fraud 🔺)

---

## Techniques Used

| Technique / Tool           | Purpose                                                            |
|---------------------------|--------------------------------------------------------------------|
| **Decision Trees**        | Supervised model to classify transactions                          |
| **SMOTE Oversampling**    | Generate synthetic fraud samples to rebalance classes              |
| **Confusion Matrix**      | Evaluate model precision, recall, F1-score                         |
| **ROC Curve + AUC**       | Visualise classification performance under various thresholds      |
| **Data Resampling**       | Ensure model doesn’t overfit to majority class                     |
| **Python (scikit-learn)** | Model building and evaluation                                      |
| **Matplotlib / Seaborn**  | Visualisation of performance metrics                               |

---

## 🔧 Implementation Steps

1. **Preprocessing**
   - Cleaned and scaled features
   - Split dataset into train/test sets

2. **Resampling with SMOTE**
   - Applied SMOTE to generate synthetic fraud cases
   - Balanced training set to improve sensitivity

3. **Model Training**
   - Trained baseline Decision Tree classifier
   - Compared performance with and without SMOTE

4. **Evaluation**
   - Confusion Matrix
   - Recall, Precision, F1-Score
   - ROC Curve and AUC

---

## Results

**True Positive Rate improved by 11%** after using SMOTE.

---

## Key Learnings

- **Recall is critical** in fraud detection: better to flag a few extra legit transactions than miss a fraud.
- **SMOTE helps balance** without discarding legit data (unlike undersampling).
- **Model performance must be assessed holistically** (AUC + recall + precision, not just accuracy).



