# Healthcare Cost Prediction

Predict annual healthcare costs using patient data (age, BMI, smoking status, region, number of children). Built with XGBoost + FastAPI.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt


# Download dataset
Download insurance.csv from Kaggle - Medical Cost Personal Dataset and place it in the data/ folder.


# Train the model
```bash
python -m src.train



# Run the API server
uvicorn api.app:app --reload --port 8000
