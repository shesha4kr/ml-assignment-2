from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# define features with numerical and categorical values
numerical_cols = [ 'age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak' ]

categorical_cols = [ 'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina', 'slope', 'vessels_colored_by_flourosopy', 'thalassemia' ]

# define preprocessor to scale and encode the features and split the dataset
def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols)
        ]
    )
