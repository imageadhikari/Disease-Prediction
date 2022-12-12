const {Paragraph, Document, Packer} = require('docx');
const fs = require('fs');

const {PatientDetails, LineBreak, CustomText, MainHeading, HeaderImage,
    ParagraphBreak} = require('./documentComponents');


const CreateDocx = (obj) => {
    const fileName = obj.PatientId;
    const doc = new Document({
        compatibility:{
            version: 17,
            suppressTopSpacing: true
        },
        sections: [
            {
                properties: {}, 
                children : [
                    new Paragraph({
                        children : [ 
                            HeaderImage(),
                        ],                                    
                    }),
                    MainHeading(),
                    new Paragraph({
                        children : [ 
                            LineBreak(2),
                            PatientDetails('Symptom1', obj.Symptom1),
                            PatientDetails('Symptom2', obj.Symptom2),
                            PatientDetails('Symptom3', obj.Symptom3),
                            PatientDetails('Symptom4', obj.Symptom4),
                            PatientDetails('Symptom5', obj.Symptom5),
                            PatientDetails('Symptom6', obj.Symptom6),
                            PatientDetails('Symptom7', obj.Symptom7),
                            PatientDetails('Symptom8', obj.Symptom8),
                            PatientDetails('Symptom9', obj.Symptom9),
                            PatientDetails('Symptom10', obj.Symptom10),
                            PatientDetails('Symptom11', obj.Symptom11),
                            PatientDetails('Symptom12', obj.Symptom12),
                            PatientDetails('Symptom13', obj.Symptom13),
                            PatientDetails('Symptom14', obj.Symptom14),
                            PatientDetails('Symptom15', obj.Symptom15),
                            PatientDetails('Symptom16', obj.Symptom16),
                            PatientDetails('Symptom17', obj.Symptom17),
                            PatientDetails(`Disease`, obj.Disease)

                        ],                                    
                    }),
                    
                ]            
            }
        ]
    })
    
    Packer.toBuffer(doc).then((Buffer) => {
        fs.writeFileSync(`./PreventionDocx/${fileName}.docx`, Buffer)
    })
}

module.exports = CreateDocx