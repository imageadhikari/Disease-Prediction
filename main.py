from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from computation import give_weight

# Training and dumping the model before loading it because Gradient Boosting Classifier struggles with pickling.
# Some attrubutes are usually left out and _loss is typically left out.
df_processed = pd.read_csv("data/processed.csv")

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

    try:
        if Symptom_1=="itching":
            inp_arr[0]=1
        elif Symptom_1=="skin rash":
            inp_arr[0]=3
        elif Symptom_1=="nodal skin eruptions":
            inp_arr[0]=4
        elif Symptom_1=="continuous sneezing":
            inp_arr[0]=4
        elif Symptom_1=="shivering":
            inp_arr[0]=5
        elif Symptom_1=="chills":
            inp_arr[0]=3
        elif Symptom_1=="joint pain":
            inp_arr[0]=3
        elif Symptom_1=="stomach pain":
            inp_arr[0]=5
        elif Symptom_1=="acidity":
            inp_arr[0]=3
        elif Symptom_1=="ulcers on tongue":
            inp_arr[0]=4
        elif Symptom_1=="muscle wasting":
            inp_arr[0]=3
        elif Symptom_1=="vomiting":
            inp_arr[0]=5
        elif Symptom_1=="burning micturition":
            inp_arr[0]=6
        elif Symptom_1=="spotting urination":
            inp_arr[0]=6
        elif Symptom_1=="fatigue":
            inp_arr[0]=4
        elif Symptom_1=="weight gain":
            inp_arr[0]=3
        elif Symptom_1=="anxiety":
            inp_arr[0]=4
        elif Symptom_1=="cold hands and feets":
            inp_arr[0]=5
        elif Symptom_1=="mood swings":
            inp_arr[0]=3
        elif Symptom_1=="weight loss":
            inp_arr[0]=3
        elif Symptom_1=="restlessness":
            inp_arr[0]=5
        elif Symptom_1=="lethargy":
            inp_arr[0]=2
        elif Symptom_1=="patches in throat":
            inp_arr[0]=6
        elif Symptom_1=="irregular sugar level":
            inp_arr[0]=5
        elif Symptom_1=="cough":
            inp_arr[0]=4
        elif Symptom_1=="high fever":
            inp_arr[0]=7
        elif Symptom_1=="sunken eyes":
            inp_arr[0]=3
        elif Symptom_1=="breathlessness":
            inp_arr[0]=4
        elif Symptom_1=="sweating":
            inp_arr[0]=3
        elif Symptom_1=="dehydration":
            inp_arr[0]=4
        elif Symptom_1=="indigestion":
            inp_arr[0]=5
        elif Symptom_1=="headache":
            inp_arr[0]=3
        elif Symptom_1=="yellowish skin":
            inp_arr[0]=3
        elif Symptom_1=="dark urine":
            inp_arr[0]=4
        elif Symptom_1=="nausea":
            inp_arr[0]=5
        elif Symptom_1=="loss of appetite":
            inp_arr[0]=4
        elif Symptom_1=="pain behind the eyes":
            inp_arr[0]=4
        elif Symptom_1=="back pain":
            inp_arr[0]=3
        elif Symptom_1=="constipation":
            inp_arr[0]=4
        elif Symptom_1=="abdominal pain":
            inp_arr[0]=4
        elif Symptom_1=="diarrhoea":
            inp_arr[0]=6
        elif Symptom_1=="mild fever":
            inp_arr[0]=5
        elif Symptom_1=="yellow urine":
            inp_arr[0]=4
        elif Symptom_1=="yellowing of eyes":
            inp_arr[0]=4
        elif Symptom_1=="acute liver failure":
            inp_arr[0]=6
        elif Symptom_1=="fluid overload":
            inp_arr[0]=6
        elif Symptom_1=="swelling of stomach":
            inp_arr[0]=7
        elif Symptom_1=="swelled lymph nodes":
            inp_arr[0]=6
        elif Symptom_1=="malaise":
            inp_arr[0]=6
        elif Symptom_1=="blurred and distorted vision":
            inp_arr[0]=5
        elif Symptom_1=="phlegm":
            inp_arr[0]=5
        elif Symptom_1=="throat irritation":
            inp_arr[0]=4
        elif Symptom_1=="redness of eyes":
            inp_arr[0]=5
        elif Symptom_1=="sinus pressure":
            inp_arr[0]=4
        elif Symptom_1=="runny nose":
            inp_arr[0]=5
        elif Symptom_1=="congestion":
            inp_arr[0]=5
        elif Symptom_1=="chest pain":
            inp_arr[0]=7
        elif Symptom_1=="weakness in limbs":
            inp_arr[0]=7
        elif Symptom_1=="fast heart rate":
            inp_arr[0]=5
        elif Symptom_1=="pain during bowel movements":
            inp_arr[0]=5
        elif Symptom_1=="pain in anal region":
            inp_arr[0]=6
        elif Symptom_1=="bloody stool":
            inp_arr[0]=5
        elif Symptom_1=="irritation in anus":
            inp_arr[0]=6
        elif Symptom_1=="neck pain":
            inp_arr[0]=5
        elif Symptom_1=="dizziness":
            inp_arr[0]=4
        elif Symptom_1=="cramps":
            inp_arr[0]=4
        elif Symptom_1=="bruising":
            inp_arr[0]=4
        elif Symptom_1=="obesity":
            inp_arr[0]=4
        elif Symptom_1=="swollen legs":
            inp_arr[0]=5
        elif Symptom_1=="swollen blood vessels":
            inp_arr[0]=5
        elif Symptom_1=="puffy face and eyes":
            inp_arr[0]=5
        elif Symptom_1=="enlarged thyroid":
            inp_arr[0]=6
        elif Symptom_1=="brittle nails":
            inp_arr[0]=5
        elif Symptom_1=="swollen extremeties":
            inp_arr[0]=5
        elif Symptom_1=="excessive hunger":
            inp_arr[0]=4
        elif Symptom_1=="extra marital contacts":
            inp_arr[0]=5
        elif Symptom_1=="drying and tingling lips":
            inp_arr[0]=4
        elif Symptom_1=="slurred speech":
            inp_arr[0]=4
        elif Symptom_1=="knee pain":
            inp_arr[0]=3
        elif Symptom_1=="hip joint pain":
            inp_arr[0]=2
        elif Symptom_1=="muscle weakness":
            inp_arr[0]=2
        elif Symptom_1=="stiff neck":
            inp_arr[0]=4
        elif Symptom_1=="swelling joints":
            inp_arr[0]=5
        elif Symptom_1=="movement stiffness":
            inp_arr[0]=5
        elif Symptom_1=="spinning movements":
            inp_arr[0]=6
        elif Symptom_1=="loss of balance":
            inp_arr[0]=4
        elif Symptom_1=="unsteadiness":
            inp_arr[0]=4
        elif Symptom_1=="weakness of one body side":
            inp_arr[0]=4
        elif Symptom_1=="loss of smell":
            inp_arr[0]=3
        elif Symptom_1=="bladder discomfort":
            inp_arr[0]=4
        elif Symptom_1=="foul smell ofurine":
            inp_arr[0]=5
        elif Symptom_1=="continuous feel of urine":
            inp_arr[0]=6
        elif Symptom_1=="passage of gases":
            inp_arr[0]=5
        elif Symptom_1=="internal itching":
            inp_arr[0]=4
        elif Symptom_1=="toxic look (typhos)":
            inp_arr[0]=5
        elif Symptom_1=="depression":
            inp_arr[0]=
        elif Symptom_1=="irritability":
            inp_arr[0]=
        elif Symptom_1=="muscle pain":
            inp_arr[0]=
        elif Symptom_1=="altered sensorium":
            inp_arr[0]=
        elif Symptom_1=="red spots over body":
            inp_arr[0]=
        elif Symptom_1=="belly pain":
            inp_arr[0]=
        elif Symptom_1=="abnormal menstruation":
            inp_arr[0]=
        elif Symptom_1=="dischromic patches":
            inp_arr[0]=
        elif Symptom_1=="watering from eyes":
            inp_arr[0]=
        elif Symptom_1=="":
            inp_arr[0]=
        elif Symptom_1=="":
            inp_arr[0]=
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



        

