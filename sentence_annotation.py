import spacy
import os
import pdfx
import re
from googleapiclient.discovery import build
from google.oauth2 import service_account

def process_document(title, source_path,source, sheet, SAMPLE_SPREADSHEET_ID):

    # READING AND MANIPULATING INPUT FILE
    path = 'data/'+source_path+'/'+title+'.pdf'
    input_file = pdfx.PDFx(path) # TO DO: OPTIMIZE PATH, GET IT STRAIGHT FROM PARAMETER INSTEAD OF CALCULATING IT AGAIN
    input_file = input_file.get_text()

    doc = nlp(input_file)

    values = []

    for span in doc.sents:
        sentence = []
        #sent = re.sub("\n", " ", str(span)) # to get DATAPOLICY3  format comment this line, and add str casting to append
        #span = re.sub("\n\n", " ", str(span))
        sentence.append(str(span))
        values.append(sentence)

    value_input_option = 'USER_ENTERED'
    sentences = {
        'values': values
    }

    sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='Test'+'!A2:A2000',valueInputOption=value_input_option, body=sentences).execute()#title


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'google_key.json'

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# The ID and range of a sample spreadsheet.
#SAMPLE_SPREADSHEET_ID = '12mT4Fl9t3UVW8Jx8NjA8SJPVWDNH0lnkUgb6cM3ZiyQ'
SAMPLE_SPREADSHEET_ID = '1trg0bot87WtOALsxiiEIVYX6VW6mIBr90GrsY-t2jRw'
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

nlp = spacy.load('en_core_web_lg') # TRY REPRODUCING WITH SM
print ("Pipeline:", nlp. pipe_names) 
nlp.enable_pipe("parser")
print ("Pipeline:", nlp. pipe_names) 

path='Privacy/Facebook/TargetCompanySourced' # TO ADD DIFFERENT DOCUMENTS, REMOVE PART AFTER LAST /
source='TargetCompanySourced'

process_document('DataPolicy', path, source, sheet, SAMPLE_SPREADSHEET_ID)

#for filename in os.listdir('data/'+path):
#    print(filename)
#    file_name, file_extension = os.path.splitext(filename)
#    process_document(file_name, path, source, sheet, SAMPLE_SPREADSHEET_ID)