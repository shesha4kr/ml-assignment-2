import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="My First Streamlit App", layout="centered")

st.title("🫀 Heart Disease Risk Prediction")
st.subheader("Different ML Model Performance Comparison")

# define features with numerical and categorical values
numerical_cols = [ 'age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak' ]
categorical_cols = [ 'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia' ]

# load CSV file uploaded in google drive
@st.cache_data
def load_data():
    file_id = "1AnHrT9-13EI9lJgCg-6i_A-YQVP2dTka"
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    df = pd.read_csv(url)
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# define preprocessor to scale and encode the features and split the dataset
@st.cache_data
def prepare_data(df):
    TARGET_COLUMN = "target"
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    # Train-test split (VERY IMPORTANT: stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor

df = load_data()
X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

# Dropdowns
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

# dropdown dispatcher
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