# Healthcare Cost Prediction

Predict annual healthcare costs using patient data (age, BMI, smoking status, region, number of children). Built with XGBoost + FastAPI.

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset
Download insurance.csv from Kaggle - Medical Cost Personal Dataset and place it in the data/ folder.


### 3. Train the model
```bash
python -m src.train
```


### 4. Run the API server
```bash
uvicorn api.app:app --reload --port 8000
```
