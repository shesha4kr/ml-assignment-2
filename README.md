## 1. Problem Statement

Build a Machine Learning model that predicts whether a heart failure patient will experience a death event (DEATH_EVENT) based on their clinical and demographic data. The goal is to help identify high-risk patients early so timely medical attention and monitoring can be prioritized.

This application evaluates multiple classification models to determine the outcome:
- **0 → No Death Event (Survived)**
- **1 → Death Event (Not Survived)**


## 2. Dataset Description

This project uses the Heart Failure Clinical Records dataset, which contains real-world clinical information of patients diagnosed with heart failure. The dataset includes demographic details, existing medical conditions, and important lab/clinical measurements such as ejection fraction, serum creatinine, and serum sodium. These attributes are highly useful for predicting a patient’s risk of mortality during the follow-up period.

### Dataset Source
The dataset was collected from Kaggle:
https://www.kaggle.com/datasets/aadarshvelu/heart-failure-prediction-clinical-records

### Dataset Overview
- **Total Records:** 5000 
- **Total Columns:** 13
- **Input Features:** 12
- **Target Column:** `target`

### Attribute Details
- **age**: age of the patient (years)
- **anaemia**: decrease of red blood cells or hemoglobin (boolean)
- **creatinine phosphokinase (CPK)**: level of the CPK enzyme in the blood (mcg/L)
- **diabetes**: if the patient has diabetes (boolean) 
- **ejection fraction**: percentage of blood leaving the heart at each contraction (percentage)  
- **high blood pressure**: if the patient has hypertension (boolean)
- **platelets**: platelets in the blood (kiloplatelets/mL)
- **sex**: woman or man (binary)
- **serum creatinine**: level of serum creatinine in the blood (mg/dL) 
- **serum sodium**: level of serum sodium in the blood (mEq/L) 
- **smoking**: woman or man (binary)
- **time**: level of serum creatinine in the blood (mg/dL) 
- **DEATH_EVENT**: if the patient died during the follow-up period (boolean)


## 3. Models Used:

## Model Performance Comparison

| ML Model Name | Accuracy (%) | AUC | Precision | Recall | F1 Score | MCC |
|-------------|--------------|-----|----------|--------|---------|-----|
| Logistic Regression | 0.856 | 0.886 | 0.797 | 0.696 | 0.743 | 0.647 |
| Decision Tree Classifier | 0.951 | 0.947 | 0.902 | 0.937 | 0.919 | 0.884 |
| K-Nearest Neighbor Classifier | 0.879 | 0.937 | 0.885 | 0.684 | 0.771 | 0.701 |
| Naive Bayes (Gaussian) | 0.799 | 0.850 | 0.703 | 0.570 | 0.629 | 0.499 |
| Random Forest | 0.966 | 0.987 | 0.961 | 0.924 | 0.942 | 0.918 |
| XGBoost | 0.955 | 0.984 | 0.947 | 0.899 | 0.922 | 0.891 |


## Observations on the Performnace

| ML Model Name | Observation about model performance |
|-------------|-----------------------------------|
| Logistic Regression | Strong baseline performance with good accuracy (0.856) and AUC (0.886). Precision is decent (0.797), but recall (0.696) is moderate, meaning it may miss some positive (death event) cases. |
| Decision Tree Classifier | Very high accuracy (0.951) and strong recall (0.937), indicating good detection of positive cases. However, single decision trees may overfit and might not generalize as consistently as ensemble models. |
| K-Nearest Neighbor Classifier | Shows good overall performance with high AUC (0.937) and strong precision (0.885). Recall (0.684) is lower, suggesting it performs better at avoiding false positives than capturing all positives. |
| Naive Bayes (Gaussian) | Lowest-performing model among the tested approaches (accuracy 0.799, F1 0.629). The independence assumption likely limits performance on this clinical dataset. |
| Random Forest | Best overall performer with the highest accuracy (0.966), AUC (0.987), F1 score (0.942), and MCC (0.918). Provides the most balanced and reliable classification results. |
| XGBoost | Excellent performance with high accuracy (0.955) and AUC (0.984). Precision (0.947) and recall (0.899) are well-balanced, making it a strong alternative to Random Forest. |

