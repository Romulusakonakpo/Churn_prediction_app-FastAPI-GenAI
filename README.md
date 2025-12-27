# Customer Churn Prediction – Telecom

This project implements a machine learning pipeline to predict customer churn in a telecom context.For this project, the TELCO dataset simulated by IBM is used.

## Project overview
- Binary churn prediction using a Random Forest classifier
- Feature engineering and preprocessing with scikit-learn
- REST API built with FastAPI
- Optional LLM-based explanation layer (Ollama)
- Streamlit front-end for interaction

## Selected features
The final model which is `RandomForestClassifier` uses the five most important features:
- `InternetService`
- `Contract`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

This choice reduces bias introduced by missing or imputed variables while maintaining strong predictive performance (`AUC = 0.83` instead 0.84 by including all variables).

## How to run

### 0. Install the requirements
```bash
pip install -r requirements
```
### 1. Train the model
```bash
python src/train.py
```
### 2. Launch docker container
```bash
docker compose up --build
```
### 3. Launch the streamlit app
```bash
streamlit run streamlit_app.py
```
