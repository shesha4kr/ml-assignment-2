from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# define features with numerical and categorical values
numerical_cols = [ 'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time' ]

categorical_cols = [ 'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# define preprocessor to scale and encode the features and split the dataset
def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             categorical_cols)
        ]
    )
