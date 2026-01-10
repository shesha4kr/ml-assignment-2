import streamlit as st

st.set_page_config(page_title="My First Streamlit App", layout="centered")

st.title("🫀 Heart Disease Risk Prediction")
st.subheader("Different ML Model Performance Comparison")


model_choice = st.selectbox(
    "Select a Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier - Gaussian",
        "Ensemble Model - Random Forest",
        "Ensemble Model - XGBoost"
    )
)

def logistic_regression():
    st.subheader("Logistic Regression Classifier")
    st.write("Running Logistic Regression model...")

def decision_tree():
    st.subheader("Decision Tree Classifier")
    st.write("Running Decision Tree Classifier...")

def knn_classifier():
    st.subheader("K-Nearest Neighbor Classifier")
    st.write("Running KNN Classifier...")

def naive_bayes():
    st.subheader("Naive Bayes Classifier")
    st.write("Running Naive Bayes (Gaussian / Multinomial)...")

def random_forest():
    st.subheader("Random Forest Classifier")
    st.write("Running Random Forest model...")

def xgboost_model():
    st.subheader("XGBoost Classifier")
    st.write("Running XGBoost model...")

# ----------------------------
# Dispatcher (Clean & Scalable)
# ----------------------------

model_dispatcher = {
    "Logistic Regression": logistic_regression,
    "Decision Tree Classifier": decision_tree,
    "K-Nearest Neighbor Classifier": knn_classifier,
    "Naive Bayes Classifier - Gaussian": naive_bayes,
    "Ensemble Model - Random Forest": random_forest,
    "Ensemble Model - XGBoost": xgboost_model
}

# Call selected model
model_dispatcher[model_choice]()