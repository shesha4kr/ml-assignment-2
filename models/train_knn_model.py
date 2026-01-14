import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
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
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "saved_models/knn_model.pkl")
print("✅ KNN classifier saved successfully")
