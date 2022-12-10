
# Disease Predictor

Disease Predictor is a webapp for predicting disease based on your symptoms. 

## What It Does

Disease Predictor is a smart app that is based on the Gradient Boosting Classifier trained using the [Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset?resource=download&select=symptom_precaution.csv) dataset from Kaggle. It is able to predict 41 different diseases based on 131 unique symptoms and their weighted relevancy for the disease. 


## Run Locally

Clone the project

```bash
  git clone https://github.com/imageadhikari/Disease-Prediction.git
```

Navigate to the project directory

```bash
  cd Disease-Prediction
```

Create a virtual environment

```bash
  python -m venv disease_env
```

Activate the environment

```bash
  disease_env\Scripts\activate.bat
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  uvicorn app:app --reload
```

