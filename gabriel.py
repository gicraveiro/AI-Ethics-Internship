#author: gabriel roccabruna
from googleapiclient.discovery import build
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'google_key.json'

creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# If modifying these scopes, delete the file token.json.


# The ID and range of a sample spreadsheet.
#SAMPLE_SPREADSHEET_ID = '1sbXZlG1egk9DpTj6uHVCqIa_sdpFdg14_dWLHYAFnv0'
SAMPLE_SPREADSHEET_ID = '12mT4Fl9t3UVW8Jx8NjA8SJPVWDNH0lnkUgb6cM3ZiyQ'
service = build('sheets', 'v4', credentials=creds)
sheet = service.spreadsheets()

result = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='DataPolicy!1:1000').execute()
print(result.get('values', []))

# author: giovana