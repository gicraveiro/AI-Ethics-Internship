import spacy
import os
import pdfx

#author: gabriel roccabruna
from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'google_key.json'

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# If modifying these scopes, delete the file token.json.

# The ID and range of a sample spreadsheet.
SAMPLE_SPREADSHEET_ID = '12mT4Fl9t3UVW8Jx8NjA8SJPVWDNH0lnkUgb6cM3ZiyQ'
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='DataPolicy!A1:A1000').execute()
values = result.get('values')

#for row in values:
#    print(row)
print(type(values))
datatest = [['linha1'],['linha2']]
print(type(datatest))

value_input_option = 'USER_ENTERED'
sentences = {
    'values': datatest
}

sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='Teste!A1:A1000',valueInputOption=value_input_option, body=sentences).execute()


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

#path='Facebook/Privacy/TargetCompanySourced'
#source='TargetCompanySourced'

#for filename in os.listdir('data/'+path):
#    print(filename)
#    file_name, file_extension = os.path.splitext(filename)
#    process_document(file_name, path, source)