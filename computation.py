severity = {}

with open("data/Symptom-severity.csv") as f:
    for line in f.read().splitlines()[1:]:
        symp, sever = line.split(",")
        severity[symp] = sever


def give_weight(symp):
    if symp and symp in severity:
        return severity[symp]

    return 0
