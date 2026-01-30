import requests

OLLAMA_URL = "http://ollama:11434/api/generate"
MODEL = "qwen2.5:1.5b"

def explain_churn(churn_proba, features: dict) -> str:
    prompt = f"""
You are a senior marketing analyst.

This client has these features:
{features}

Churn probability: {churn_proba:.2f}

Feature meanings:
- InternetService: type of internet the client uses (e.g., "Fiber optic", "DSL", "No")
- Contract: length of contract (e.g., "Month-to-month", "One year", "Two year")
- tenure: number of months as a client
- MonthlyCharges: current monthly bill
- TotalCharges: total bill paid so far

Rules:
- Refer to this person using demonstrative pronouns/determiners only ("this client", "these choices")
- Do not speak about customers in general
- Base reasoning only on provided features
- Be concrete, factual, and specific
- If churn <0.5: mention low risk and propose 2-3 concrete actions to improve experience
- If churn >=0.5: explain risk and suggest 2-3 retention actions
- Direct, personalized, professional, one short paragraph

Examples:
1. "This client presents a low churn probability. Given this contract, monthly and total charges, improving their experience could include upgrading service options and offering personalized guidance."
2. "This client shows high churn risk. Because of this short tenure and month-to-month contract, retention actions include offering incentives and personalized support."
"""

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 120,
            "top_p": 0.8
        }   
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)

        if response.status_code != 200:
            return f"Try again please ! Error : {response.status_code}: {response.text}"

        data = response.json()

        if "response" not in data:
            return f"Unexpected LLM output: {data}"

        return data["response"]

    except requests.exceptions.RequestException:
        # Message conditionnel selon le churn probability
        risk_phrase = (
            "This client presents a low risk of churn."
            if churn_proba < 0.5
            else "This client presents a high risk of churn."
        )
        return (
            f"{risk_phrase} Prediction was generated successfully, "
            "but the explanation service is temporarily unavailable."
        )
