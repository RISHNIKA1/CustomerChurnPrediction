import joblib
import pandas as pd


def load_model(path="/Users/balavishnu/Desktop/Datascience/CustomerChurn_Prediction/models/model.pkl"):
    return joblib.load(path)


def prdict_churn(model, input_dict):
    churn_data = pd.DataFrame([input_dict])
    prediction = model.predict(churn_data)[0]
    proba = model.predict_proba(churn_data)[0][1]
    
    return prediction, proba
