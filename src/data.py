import os
import numpy as np
import pandas as pd
import kagglehub

def load_data() -> pd.DataFrame:
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    file_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Type casting
    dtypes = {
        'gender': 'category',
        'SeniorCitizen': 'category',
        'Partner': 'category',
        'Dependents': 'category',
        'PhoneService': 'category',
        'MultipleLines': 'category',
        'InternetService': 'category',
        'OnlineSecurity': 'category',
        'OnlineBackup': 'category',
        'DeviceProtection': 'category',
        'TechSupport': 'category',
        'StreamingTV': 'category',
        'StreamingMovies': 'category',
        'Contract': 'category',
        'PaperlessBilling': 'category',
        'PaymentMethod': 'category'
    }
    df = df.astype(dtypes)

    # TotalCharges fix
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'].replace('', np.nan),
        errors='coerce'
    )
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    return df
