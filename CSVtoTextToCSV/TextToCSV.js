const fs = require('fs');
const ArrayToCSV = require('convert-array-to-csv');

let PrecautionsTxt = "./PrecautionsText/";
let SymptomsTxt = "./SymptomsText";


const divideTxtFile = (fileName, query) => {
    const contents = fs.readFileSync(`${query}/${fileName}`, 'utf-8').split('\n')
    let fileData = [];
    let result = [];
    for (let row of contents) {
        let rowSplit = row.split(':')
        fileData.push(rowSplit)
    }
    if(query == SymptomsTxt){
        fileData[0] = fileData[18]
        fileData = fileData.slice(0, 18)    
    }
    for (let data of fileData) {
        result.push(data[1])
    }
    return result
}

const getSymptomsCSV = () => {
    const finalData = []
    const header = ['Disease','Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5',
                'Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10','Symptom_11',
                'Symptom_12','Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17'];
    fs.readdir(SymptomsTxt, (err, files) => {
        if (err)
            throw err
        for (let fileName of files){
            finalData.push(divideTxtFile(`${fileName}`, SymptomsTxt))
        }
        const arraytocsv = ArrayToCSV.convertArrayToCSV(finalData,{
            header,
            separator: ','
        })
        fs.writeFile(`./Results/Symptoms.csv`, arraytocsv, (err) => {
            throw err
        })
    })
}

const getPrecautionsCSV = () => {
    const finalData = []
    const header = ['Disease','Precaution_1','Precaution_2','Precaution_3','Precaution_4'];
    fs.readdir(PrecautionsTxt, (err, files) => {
        if (err)
            throw err
        for (let fileName of files){
            finalData.push(divideTxtFile(`${fileName}`, PrecautionsTxt))
        }
        const arraytocsv = ArrayToCSV.convertArrayToCSV(finalData,{
            header,
            separator: ','
        })
        fs.writeFile(`./Results/Precautions.csv`, arraytocsv, (err) => {
            throw err
        })
    })
}

// getSymptomsCSV()
getPrecautionsCSV()







