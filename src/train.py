import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocesss import load_data,split_data,build_preprocessor


def train_model(data_path="/Users/balavishnu/Desktop/Datascience/CustomerChurn_Prediction/data/Customer_Churn_dataset.csv"):
    churn_data =load_data(data_path)
    
    # fix total charged issue
    if "TotalCharges" in churn_data.columns:
        churn_data["TotalCharges"] = pd.to_numeric(churn_data["TotalCharges"],errors="coerce")
        churn_data["TotalCharges"] = churn_data["TotalCharges"].fillna(churn_data["TotalCharges"].median())
        
    
    X_train, X_test, y_train, y_test = split_data(churn_data)
    preprocessor = build_preprocessor(X_train)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth= 10,
        random_state= 42
    )
    
    clf = Pipeline(steps =[
        ('preprocessor', preprocessor),
        ('model',model)
    ])
    
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    
    print("Classification Report :\n")
    print(classification_report(y_test,y_pred))
    
    print("ROC-AUC score :", roc_auc_score(y_test,y_proba))
    
    joblib.dump(clf, "/Users/balavishnu/Desktop/Datascience/CustomerChurn_Prediction/models/model.pkl")
    print("\n Model saved as models/model.pkl")
    
    
if __name__ == "__main__":
    train_model()