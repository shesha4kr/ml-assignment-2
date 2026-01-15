import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

from preprocessing import get_preprocessor

df = pd.read_csv("data/heart_failure_clinical_records.csv")
# Remove duplicates based only on feature columns
feature_cols = df.columns.drop("DEATH_EVENT")
df = df.drop_duplicates(subset=feature_cols).reset_index(drop=True)
X = df.drop(columns=["DEATH_EVENT"])
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        ))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "saved_models/xgboost_model.pkl")
print("✅ XGBoost model saved successfully")
