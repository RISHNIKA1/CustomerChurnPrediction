import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path):
    churn_data = pd.read_csv(path)
    return churn_data

def build_preprocessor(X):
    num_cols = X.select_dtypes(include =["int64","float64"]).columns
    cat_cols = X.select_dtypes(include = ["object"]).columns
    
    numeric_transformer = Pipeline(steps=[
        ("scalar", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps =[
        ("onehot",OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers =[
            ("num", numeric_transformer,num_cols),
            ("cat", categorical_transformer,cat_cols)
        ]
    )
    
    return preprocessor

def split_data(churn_data,target ="Churn"):
    
    if "customerID" in churn_data.columns:
        churn_data = churn_data.drop(columns = ["customerID"])
        
        
    X = churn_data.drop(columns=[target])
    y = churn_data[target].map({"Yes" :1, "No" :0})
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify =y )