from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    num_features = X.select_dtypes(include=['int', 'float']).columns.tolist()
    cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_features),
        ('cat', categorical_pipeline, cat_features)
    ])

    return preprocessor, num_features, cat_features
