import pandas as pd

df_sympSeverity = pd.read_csv("data/Symptom-severity.csv")

def give_weight(w):
    for i in range(df_sympSeverity.shape[0]):
        if df_sympSeverity['Symptom'][i]==w:
            return df_sympSeverity['weight'][i]
        # else:
        #     return 0