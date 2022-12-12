const fs = require('fs');
const readLine = require('readline');
const Papa = require('papaparse');
const CreateDocx = require('./docxTemplate');

const SymptomStream = fs.createReadStream('.DataSets/Dataset.csv');
const SymptomReader = readLine.createInterface({input: SymptomStream});

let SymptomList = [];


let obj = {};
const Temp = [];

SymptomReader.on("line", row => {
    SymptomList.push(row.split(','));
})

SymptomReader.on("close", ()=> {
    let PatientId = 0
    for (let iS = 1; iS< SymptomList.length; iS++){
        for (let jS = 0; jS < SymptomList[iS].length; jS++){
            ++PatientId;
            obj = {
                    "PatientId": PatientId/18,
                    "Symptom1": SymptomList[iS][1],
                    "Symptom2": SymptomList[iS][2],
                    "Symptom3": SymptomList[iS][3],
                    "Symptom4": SymptomList[iS][4],
                    "Symptom5": SymptomList[iS][5],
                    "Symptom6": SymptomList[iS][6],
                    "Symptom7": SymptomList[iS][7],
                    "Symptom8": SymptomList[iS][8],
                    "Symptom9": SymptomList[iS][9],
                    "Symptom10": SymptomList[iS][10],
                    "Symptom11": SymptomList[iS][11],
                    "Symptom12": SymptomList[iS][12],
                    "Symptom13": SymptomList[iS][13],
                    "Symptom14": SymptomList[iS][14],
                    "Symptom15": SymptomList[iS][15],
                    "Symptom16": SymptomList[iS][16],
                    "Symptom17": SymptomList[iS][17],                
                    "Disease": SymptomList[iS][0]            
                };                
            }
            Temp.push(obj)
            CreateDocx(obj)

        // for (let i in Temp){
        //     console.log(Temp[i].patientId)
        //     CreateFile(Temp[i])
        // }
    }
})

