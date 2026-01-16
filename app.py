import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,   confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Heart Failure Risk Predictor", layout="centered")

st.title("🫀 Heart Disease Risk Prediction")
st.subheader("Different ML Model Performance Comparison")

st.markdown("<br>", unsafe_allow_html=True) 

# upload test.csv file and download options
col1, col2 = st.columns([2.5, 1.5])

with col1:
    uploaded_file = st.file_uploader("Upload a CSV file to evaluate the model. Don’t have one? Download a sample by clicking on 'Download Test CSV'.", type=None)

with col2:

    file_path = "data/test_data_without_target.csv"

    with open(file_path, "rb") as f:
        csv_data = f.read()

    st.download_button(
        label="⬇️ Download Test CSV",
        data=csv_data,
        file_name="test_dataset_without_target.csv",
        mime="text/csv"
    )


# if file is uploaded, show dropdowns
if uploaded_file:

    if uploaded_file and not uploaded_file.name.endswith(".csv"):
        st.error("Please upload a CSV file")
        st.stop()

    if uploaded_file.size == 0:
        st.error("❌ Uploaded file is empty. Please upload a valid CSV.")
        st.stop()

    uploaded_file.seek(0)

    EXPECTED_COLUMNS = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time', 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

    df = pd.read_csv(uploaded_file)

    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing_cols or extra_cols:
        st.error("❌ Wrong file uploaded!")
        st.warning(f"Missing columns: {missing_cols}, Extra column: {extra_cols}")
        st.stop()

    @st.cache_resource
    def load_model(path):
        if not path:
            return None
        return joblib.load(path)

    model_paths = {
    "Logistic Regression": "saved_models/logistic_model.pkl",
    "Decision Tree Classifier": "saved_models/decision_tree_model.pkl",
    "K-Nearest Neighbor Classifier": "saved_models/knn_model.pkl",
    "Naive Bayes Classifier - Gaussian": "saved_models/naive_bayes_model.pkl",
    "Random Forest": "saved_models/random_forest_model.pkl",
    "XGBoost": "saved_models/xgboost_model.pkl",
    }

    st.markdown("<br>", unsafe_allow_html=True) 

    model_choice = st.selectbox("Select Model", model_paths.keys())

    model = load_model(model_paths[model_choice])

    if model is None:
        st.warning("Model not available yet.")
    else :
        X_test = df.copy() 
        df2 = pd.read_csv("data/test_target.csv") 
        y_test = df2.copy() 
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        # AUC (needs probabilities)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        st.subheader(model_choice)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("Precision", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", f"{f1:.3f}")
        col5.metric("MCC Score", f"{mcc:.3f}")
        col6.metric("AUC Score", f"{auc:.3f}" if auc is not None else "N/A")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=20)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted Label", fontsize = 5)
        ax.set_ylabel("True Label", fontsize = 5)
        ax.set_title("Confusion Matrix", fontsize = 6)

        # Reduce tick label sizes
        ax.tick_params(axis='both', labelsize=7)

        plt.tight_layout(pad=0.3)
        st.pyplot(fig)