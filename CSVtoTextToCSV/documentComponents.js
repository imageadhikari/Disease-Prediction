const {Paragraph, HeadingLevel, ImageRun, Header, TextRun, WidthType, Table} = require('docx');
const docx = require('docx');
const fs  = require('fs');

const PatientDetails = (key, value) => {
    return new TextRun({
        text : `${key}: \t${value}`,
        size: 26,
        break: 2,
        alignment: docx.AlignmentType.RIGHT,
        bold: true
    })
}

const LineBreak = (numberOfBreak) => {
    return new TextRun({
        break: parseInt(numberOfBreak)
    })
}

const ParagraphBreak = (numberOfBreak) => {
    return new Paragraph({
        children : [ 
            LineBreak(numberOfBreak)
        ],                                    
    })
}

const CustomText = (text, size = 20, isbold = false, lineBreak = 1) => {
    return new TextRun({
        text : `${text}`,
        bold: isbold,
        size: size,
        break: lineBreak
    })
}

const MainHeading = (text) => {
    return new Paragraph({
        text : "Hospital Medical Report",
        bold: true,
        heading: HeadingLevel.HEADING_1,
        alignment: docx.AlignmentType.CENTER,
        border: {
            top: {
                color: "#000000",
                space: 1,
                style: "single",
                size: 15,
            },
            bottom: {
                color: "#000000",
                space: 1,
                style: "single",
                size: 15,
            },
        },
        
    })
}

const HeaderImage = () => {
    return new docx.ImageRun({
        data: fs.readFileSync('./Header1.png'),
        transformation: {
            width: 650,
            height: 100
        },

    })
}

module.exports = {HeaderImage, PatientDetails, LineBreak, CustomText, MainHeading, ParagraphBreak}