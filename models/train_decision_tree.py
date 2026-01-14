import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from preprocessing import get_preprocessor

df = pd.read_csv("test_data/HeartDiseaseTrain-Test.csv")
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "saved_models/decision_tree_model.pkl")
print("✅ Decison Tree classifier saved successfully")
