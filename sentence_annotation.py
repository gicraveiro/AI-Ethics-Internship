import spacy
import os
import pdfx
from googleapiclient.discovery import build
from google.oauth2 import service_account

def process_document(title, source_path, sheet, SAMPLE_SPREADSHEET_ID):

    # READING AND MANIPULATING INPUT FILE
    path = 'data/'+source_path+'/'+title+'.pdf'
    input_file = pdfx.PDFx(path) 
    input_file = input_file.get_text()

    doc = nlp(input_file)

    values = []

    # Separating corpus in sentences
    for span in doc.sents:
        sentence = []
        sentence.append(str(span))
        values.append(sentence)

    # Formatting sentences to fill the spreadsheet
    value_input_option = 'USER_ENTERED'
    sentences = {
        'values': values
    }

    # Write sentences in the annotation spreadsheet
    sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range=title+'!A2:A2000',valueInputOption=value_input_option, body=sentences).execute()

# Setting up access to annotation file in google spreadsheet

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'google_key.json'

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)

SAMPLE_SPREADSHEET_ID = '1trg0bot87WtOALsxiiEIVYX6VW6mIBr90GrsY-t2jRw'
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

nlp = spacy.load('en_core_web_sm') # Obs: splitting for annotation was executed with sm but ideally should have been with large model

path='Privacy/Facebook/TargetCompanySourced' # TO ADD DIFFERENT DOCUMENTS, UPDATE PATH

# Loops through all the documents in the specified folder
for filename in os.listdir('data/'+path):
   print(filename)
   file_name, file_extension = os.path.splitext(filename)
   process_document(file_name, path, sheet, SAMPLE_SPREADSHEET_ID)