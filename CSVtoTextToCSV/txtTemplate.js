const fs = require('fs');

const CreateFile = (obj) => {
    const fileName = obj.DiseaseID;
    let patientData ='';
    for (let key in obj){
        let txtFormat = `${key} : ${obj[key]}\n`;
        patientData = patientData.concat(txtFormat)
    }
    // console.log(patientData)
    fs.writeFile(`./PrecautionsText/${fileName}.txt`, patientData, (err) => {
        throw err
    })
}
// let obj1 = {'PatientId': 10, 'Patient Name': 'TEST TEST', 'Patient Age': 90};
// let obj2 = {'PatientId': 20, 'Patient Name': 'TEST TEST', 'Patient Age': 90};
// let obj3 = {'PatientId': 30, 'Patient Name': 'TEST TEST', 'Patient Age': 90};
// let test = [obj1,obj2, obj3];

// for (let i in test){
//     console.log(test[i].PatientId)
//     CreateFile(test[i])
// }


module.exports = CreateFile