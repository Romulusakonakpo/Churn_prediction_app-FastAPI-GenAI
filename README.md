# Customer Churn Prediction – Telecom

This project implements a machine learning pipeline to predict customer churn in a telecom context.For this project, the TELCO dataset simulated by IBM is used. The dataset was automatically

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
### 3. Load the llm from ollama
```bash
docker compose exec ollama ollama pull qwen2.5:1.5b
```
### 3. Launch the streamlit app
```bash
streamlit run streamlit_app.py
```
## Usage / Notes importantes

After entering the client's information and clicking the "Predict churn" button, the system will generate a prediction and explanation.  
⚠️ Please note: Because we are using a lightweight LLM model (qwen1.5:1b) with limited computational resources, the explanation may take **around 1 minute** to be generated.  
The churn prediction itself is fast, but the explanation requires additional processing time.
