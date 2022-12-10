import pickle

import numpy as np
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates

from computation import give_weight

model = pickle.load(open("model.pkl", "rb"))
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Routes
@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def predict(request: Request):

    form = await request.form()

    success, error = False, False

    symptoms = {}
    for i in range(17):
        symptoms[f"Symptom_{i+1}"] = form.get(f"Symptom_{i+1}")

    inp_arr = np.zeros(17)
    for i in range(17):
        inp_arr[i] = give_weight(symptoms[f"Symptom_{i+1}"])

    try:
        prediction = model.predict([inp_arr])
        success = True

    except Exception as e:
        error = True
        print(e)

    context = {
        "request": request,
        "prediction": prediction,
        "error": error,
        "success": success,
    } | symptoms

    return templates.TemplateResponse("index.html", context)
