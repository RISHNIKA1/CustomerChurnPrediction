 CustomerChurnPrediction
End-to-end Customer Churn Prediction using Machine Learning (RandomForest) with Streamlit deployment.

# 📌 Project Overview
This project predicts whether a customer will churn (leave the company) using machine learning.

## 🚀 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Joblib

## 📂 Project Structure
CustomerChurnPrediction/
├── data/
├── models/
|-- notebook/
├── src/
├── app.py
├── requirements.txt
|-- .gitignore
|-- churn_env  
└── README.md


## 🔍 ML Workflow
1. Data Loading
2. Data Cleaning
3. Feature Engineering (Encoding + Scaling)
4. Model Training (RandomForest)
5. Evaluation (ROC-AUC)
6. Model Saving
7. Deployment using Streamlit

## ▶️ How to Run
### Install dependencies
```bash
pip install -r requirements.txt

## Train Model

cd src
python train.py

## Run Streamlit app
cd ..
streamlit run app.py

## Output

The Streamlit app takes customer details as input and predicts:

Churn / Not Churn

Probability Score
