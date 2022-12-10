from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
# from computation import give_weight

# Training and dumping the model before loading it because Gradient Boosting Classifier struggles with pickling.
# Some attributes are usually left out and _loss is typically left out.
df_processed = pd.read_csv("data/processed.csv")
df_sympSeverity = pd.read_csv("data/processedseverity.csv")

X = df_processed.iloc[:,1:].values
Y = df_processed['Disease'].values

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=1, stratify=Y)

GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)

pickle.dump(GBC,open('model.pkl','wb'))


model = pickle.load(open('model.pkl','rb'))
app = FastAPI()

templates = Jinja2Templates(directory="templates")
         
#Routes
@app.get("/")
def landing(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
def give_weight(w):
    for i in range(df_sympSeverity.shape[0]):
        try:
            if df_sympSeverity['Symptom'][i]==w:
                return df_sympSeverity['weight'][i]
        except:
            return 0

async def predict(request: Request):

    form = await request.form()
    # form inputs
    Symptom_1 = form.get("Symptom_1")
    Symptom_2 = form.get("Symptom_2")
    Symptom_3 = form.get("Symptom_3")
    Symptom_4 = form.get("Symptom_4")
    Symptom_5 = form.get("Symptom_5")
    Symptom_6 = form.get("Symptom_6")
    Symptom_7 = form.get("Symptom_7")
    Symptom_8 = form.get("Symptom_8")
    Symptom_9 = form.get("Symptom_9")
    Symptom_10 = form.get("Symptom_10")
    Symptom_11 = form.get("Symptom_11")
    Symptom_12 = form.get("Symptom_12")
    Symptom_13 = form.get("Symptom_13")
    Symptom_14 = form.get("Symptom_14")
    Symptom_15 = form.get("Symptom_15")
    Symptom_16 = form.get("Symptom_16")
    Symptom_17 = form.get("Symptom_17")

    inp_arr = np.zeros(17)
    success, error = False, False
    inp_arr[0]=give_weight(Symptom_1)
    inp_arr[1]=give_weight(Symptom_2)
    inp_arr[2]=give_weight(Symptom_3)
    inp_arr[3]=give_weight(Symptom_4)
    inp_arr[4]=give_weight(Symptom_5)
    inp_arr[5]=give_weight(Symptom_6)
    inp_arr[6]=give_weight(Symptom_7)
    inp_arr[7]=give_weight(Symptom_8)
    inp_arr[8]=give_weight(Symptom_9)
    inp_arr[9]=give_weight(Symptom_10)
    inp_arr[10]=give_weight(Symptom_11)
    inp_arr[11]=give_weight(Symptom_12)
    inp_arr[12]=give_weight(Symptom_13)
    inp_arr[13]=give_weight(Symptom_14)
    inp_arr[14]=give_weight(Symptom_15)
    inp_arr[15]=give_weight(Symptom_16)
    inp_arr[16]=give_weight(Symptom_17)

    try:       
        prediction = model.predict([inp_arr])

        success = True

    except:
        error = True

    return templates.TemplateResponse(
            "index.html",
            {
                "prediction": prediction,
                "error": error,
                "success": success,
                "Symptom_1": Symptom_1,
                "Symptom_2": Symptom_2,
                "Symptom_3": Symptom_3,
                "Symptom_4": Symptom_4,
                "Symptom_5": Symptom_5,
                "Symptom_6": Symptom_6,
                "Symptom_7": Symptom_7,
                "Symptom_8": Symptom_8,
                "Symptom_9": Symptom_9,
                "Symptom_10": Symptom_10,
                "Symptom_11": Symptom_11,
                "Symptom_12": Symptom_12,
                "Symptom_13": Symptom_13,
                "Symptom_14": Symptom_14,
                "Symptom_15": Symptom_15,
                "Symptom_16": Symptom_16,
                "Symptom_17": Symptom_17,
            },
        )



        

