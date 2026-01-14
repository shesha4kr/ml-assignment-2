import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocessing import get_preprocessor

file_id = "1AnHrT9-13EI9lJgCg-6i_A-YQVP2dTka"
url = f"https://drive.google.com/uc?id={file_id}&export=download"
df = pd.read_csv(url)
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline(
    steps=[
        ("preprocessor", get_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "saved_models/logistic_model.pkl")
print("✅ Logistic model saved successfully")
