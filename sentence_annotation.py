import spacy
import os
import pdfx

def process_document(title, source_path,source):
    
    # CREATING OUTPUT FILES
    sentences_path = 'output/'+source_path+'/'+title+'/Sentences.txt'
    
    os.makedirs(os.path.dirname(sentences_path), exist_ok=True)
    output_file = open(sentences_path, 'w')
    #print("\n"+title+"\n", file=output_file)
    
    # READING AND MANIPULATING INPUT FILE
    path = 'data/'+source_path+'/'+title+'.pdf'
    input_file = pdfx.PDFx(path) # TO DO: OPTIMIZE PATH, GET IT STRAIGHT FROM PARAMETER INSTEAD OF CALCULATING IT AGAIN
    input_file = input_file.get_text()

    doc = nlp(input_file)

    for span in doc.sents:
        print(span, file=output_file)

    output_file.close()

nlp = spacy.load('en_core_web_sm')

path='Facebook/Privacy/TargetCompanySourced'
source='TargetCompanySourced'

for filename in os.listdir('data/'+path):
    print(filename)
    file_name, file_extension = os.path.splitext(filename)
    process_document(file_name, path, source)