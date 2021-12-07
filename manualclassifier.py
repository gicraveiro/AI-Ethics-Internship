import json
import os
import re # regular expressions
from sklearn.metrics import precision_score, f1_score, ConfusionMatrixDisplay, recall_score # confusion_matrix, multilabel_confusion_matrix,
#from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from collections import Counter
#import csv
from googleapiclient.discovery import build
from google.oauth2 import service_account
import numpy
import copy

MAX_N_SENTENCES = 100

# Read input sentences
#path = 'output/partition/fbdata_dev.json'
path = 'output/partition/multilabeldata_dev.json'

# FLAG - CHECK IF RIGHT AND UPDATED FILE IS BEING PICKED 

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences_ref = json.loads(document)

ref_array = [sent['label'] for sent in json_sentences_ref]
ref_sent = [[sent['text']] for sent in json_sentences_ref]

counter = Counter(tuple(item) for item in ref_array)
tot_sents = len(ref_array)
# TO DO: CREATE FUNCTION TO CALCULATE THIS MAYBE?
distr_violation = round (ref_array.count(['Violate privacy'])/tot_sents, 2)
distr_commit = round (ref_array.count(['Commit to privacy'])/tot_sents, 2)
distr_related = round (ref_array.count(['Related to privacy'])/tot_sents, 2)
distr_opinion = round(ref_array.count(['Declare opinion about privacy'])/tot_sents, 2)
distr_notApp = round(ref_array.count(['Not applicable'])/tot_sents, 2)
count_violation = ref_array.count(['Violate privacy'])
count_commit = ref_array.count(['Commit to privacy'])
count_related = ref_array.count(['Related to privacy'])
count_opinion = ref_array.count(['Declare opinion about privacy'])
count_notApp = ref_array.count(['Not applicable'])

# FLAG - THERE IS PROBABLY A PYTHON FUNCTION THAT CALCULATES THIS

# Multilabel distribution chart
x_pos = numpy.arange(7) # sets number of bars
plt.bar(x_pos, counter.values(),align='center')
plt.xticks(x_pos, counter.keys(), rotation=45, ha="right") # sets labels of bars and their positions
plt.subplots_adjust(bottom=0.4) # creates space for complete label names
plt.savefig('output/Simple Classifier/multilabel_distribution.jpg')

print('Distribution of labels\nViolate privacy:', distr_violation,'\nCommit to privacy:', distr_commit,'\nOpinion about privacy:',distr_opinion, '\nRelated to privacy:', distr_related, '\nNot applicable:', distr_notApp,'\n')

# FLAG - CHECK IF DISTRIBUTION IS BEING MEASURED CORRECTLY

output_dict = []

# TO DO: RECHOOSE RULES, CHOOSE ONLY WORDS WITH SEMANTIC MEANING
# remove the words that are not in the train set even though they make sense
# verbs - infinitive, noun root form

# SIMPLE CLASSIFIER
for sentence in json_sentences_ref:
    label = [] # label = 'Not applicable'
    related = re.search(r".*data.*|.*information.*|.*cookies.*|.*personalize.*|.*content.*", sentence['text'], re.IGNORECASE)
    violate = re.search(r".*collect.*|.*share.*|.*connect.*|.*off facebook.*|.*receive information about you.*|.*\bsee\b.*", sentence['text'], re.IGNORECASE) # |.*see.*
    commit = re.search(r".*[instagram|facebook] settings.*|.*learn.*|.*choose.*|.*control.*|.*manage.*|.*opt out.*|.*delete.*|.*your consent.*|.*allow.*|.*change.*|.*choices.*|.*select.*|.*require.*|.*protection.*|.*reduce.*|.*don't sell.*", sentence['text'], re.IGNORECASE) # if 1/2 if you choose, protection(YES)law?(NO), you will have control, report(no...), will not be able, restrict, reduce, remove, don't sell, impose []restrictions, permission, you [can/will have] control, preferences(not ueful in this case), you own
    opinion = re.search(r".*should.*|.*believe.*", sentence['text'], re.IGNORECASE)

# FLAG - CHECK IF RULES DO WHAT I INTENDED

# TO DO: add weight to decide primary and secondary labels
    if violate:
        label.append('Violate privacy')
    if commit:
        label.append('Commit to privacy')
    if opinion:
        label.insert(0,'Declare opinion about privacy') # opinion related terms are stronger than commit and violate related terms
    if (related and label == []): # for single label rule, add condition: or violate and commit
        label.append('Related to privacy') # related is only activated if none of the other categories were detected
    if label == []:
        label = ['Not applicable']

    output_dict.append({"text":sentence['text'], "label":label})

# WRITE OUTPUT
path = 'output/Simple Classifier/multilabelPredictions_dev.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/multilabelPredictions_dev.json', 'w') as train_file:
    train_file.write(json.dumps(output_dict, indent=4, ensure_ascii=False))

# FLAG - CHECK IF OUTPUT WAS CORRECTLY WRITTEN IN FILE

with open(path) as file:
    document = file.read()
    json_sentences_predicted = json.loads(document)

pred_array = [sent['label'] for sent in json_sentences_predicted]

#pred_1label_simple = copy.deepcopy(pred_array)
pred_array_ordered = copy.deepcopy(pred_array) # vector of predictions that orders the prediction labels in order to align with primary label from reference with first
pred_1label_array = copy.deepcopy(pred_array) # vector of predictions that contains only first label
ref_1label_array = copy.deepcopy(ref_array) # vector of references of labels that contains only primary label
for i, (ref_label, pred_label) in enumerate(zip(ref_array, pred_array)):
    if(len(pred_label) > 1):
        pred_1label_array[i] = [pred_label[0]]
        ref_1label_array[i] = [ref_label[0]]
        #pred_1label_simple[i] = [pred_label[0]]
        if(pred_label[1] == ref_label[0]):
            pred_array_ordered[i][0] = pred_label[1]
            pred_1label_array[i] = [pred_label[1]]
            pred_array_ordered[i][1] = pred_label[0]
    elif(len(ref_label) > 1):
        ref_1label_array[i] = [ref_label[0]]
        pred_1label_array[i] = [pred_label[0]]
        #pred_1label_simple[i] = [pred_label[0]]
        if(ref_label[1] == pred_label[0]):
            #ref_1label_array[i][0] = ref_label[1]
            ref_1label_array[i] = [ref_label[1]]
            #ref_1label_array[i][1] = ref_label[0]
        
ref_1label_str_list = [label[0] for label in ref_1label_array]
pred_1label_str_list = [label[0] for label in pred_1label_array]

print(ref_1label_str_list)
print(pred_1label_str_list)

# FLAG - CHECK IF PREDICTIONS WERE CORRECTLY FILTERED TO PRIMARY LABEL

# CREATE DISTRIBUTION CHART OF ONLY 1 LABEL

# FLAG - CHECK IF PREDICTIONS ARE CORRECTLY CALCULATED - ASK GABRIEL IF THERE IS AN AUTOMATED WAY TO DO IT

ConfusionMatrixDisplay.from_predictions(ref_1label_str_list,pred_1label_str_list, normalize="true")
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.4)
#plt.show()
plt.savefig('output/Simple Classifier/remainingLabel_confusion_matrix.jpg')

# FLAG  - CHECK IF CONFUSION MATRIX IS CORRECT 
labels = sorted(list(set(ref_1label_str_list))) # sorted(list(set))

# accuracy too
precision = precision_score(ref_1label_str_list,pred_1label_str_list,labels=labels, average=None) # 
precision_micro = precision_score(ref_1label_str_list,pred_1label_str_list, average='micro')
precision_macro = precision_score(ref_1label_str_list,pred_1label_str_list, average='macro')
recall = recall_score(ref_1label_str_list,pred_1label_str_list, average=None)
recall_micro = recall_score(ref_1label_str_list,pred_1label_str_list, average='micro') # and recall macro
f1score = f1_score(ref_1label_str_list,pred_1label_str_list, average=None)
f1score_global = f1_score(ref_1label_str_list,pred_1label_str_list, average='micro')
f1score_individual = f1_score(ref_1label_str_list,pred_1label_str_list, average='macro')
f1score_weighted = f1_score(ref_1label_str_list,pred_1label_str_list, average='weighted')

# FLAG - CHECK IF CALCULATIONS ARE CORRECT

#precision = precision.tolist()
#recall = recall.tolist()
#f1score = f1score.tolist()
#print(type(labels))
#print(precision.tolist())

# CSV FILE WITH OUTPUT STATS IN IMPLEMENTATION- IS IT NECESSARY?

#with open('output/Simple Classifier/SimpleClassifierMultiLabelStats.csv', 'w') as csvfile:
    #filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    #ref_labels.insert(0, 'Classes')
    #precision.insert(0, 'Precision')
    #recall.insert(0, 'Recall')
    #f1score.insert(0, 'F1 Score')
    #print(type(labels), labels)
    #filewriter.writerow(ref_labels)
    #filewriter.writerow(precision)
    #filewriter.writerow(recall)
    #filewriter.writerow(f1score)

#values = [] # list, TO DO: ADD STUFF IN IT? CHANGE TO CSV APPROACH OF WRITING MAYBE?
#values.append(precision)
#values.append(recall)
#values.apo
#ref_sent.insert(0,'Sentences')
#ref_array.insert(0,'Correct label')
#pred_array.insert(0,'Predicted label')

# RETHINK THIS WHOLE PART BELOW

'''
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
value_input_option = 'USER_ENTERED'

stats_values = []
labels.insert(0,'Classes')
stats_titles = [['Classification Statistics'],labels,['Precision:'],['Precision micro:'],['Recall:'],['Recall micro:'],['F1 Score:'],['F1 Score micro:'],['F1 Score macro:'],['F1 Score weighted:']]
stats_values.extend([precision.tolist(), [precision_micro.tolist()], recall.tolist(), [recall_micro.tolist()], f1score.tolist(), [f1score_global.tolist()], [f1score_individual.tolist()], [f1score_weighted.tolist()]])

# CONVERTING TO FORMAT GOOGLE SHEETS ACCEPT
sentences = {
        'values': ref_sent #values
    }
ref_values = {
        'values': ref_array #values
    }
predicted_values = {
        'values': pred_array_ordered #values
    }
predictions_1label = {
        'values': pred_1label_array
}
refs_1label = {
        'values': ref_1label_array
}
stats_titles_json = {
        'values': stats_titles
}
stats_values_json = {
        'values': stats_values
}

# WRITING RESULTS IN GOOGLE SHEETS - MAYBE WRITE THEM SOMEWHERE ELSE - BTW ORGANIZE GOOGLE SHEETS
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='ManualClassifierPredictions'+'!A2:A'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=sentences).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='ManualClassifierPredictions'+'!B2:C'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=ref_values).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='ManualClassifierPredictions'+'!D2:E'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=predicted_values).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='1labelResults'+'!A2:A'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=sentences).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='1labelResults'+'!B2:B'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=refs_1label).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='1labelResults'+'!C2:C'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=predictions_1label).execute()

sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='1labelResults'+'!E1:K'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=stats_titles_json).execute()
sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='1labelResults'+'!F3:K'+str(MAX_N_SENTENCES),valueInputOption=value_input_option, body=stats_values_json).execute()

# PRINTING RESULTS IN THE TERMINAL
print("\nClassification Statistics\n")
# accuracy
print('Precision:',precision)
print('Precision:',precision_micro) # and macro
print('Recall:',recall)
print('Recall micro:',recall_micro) # and macro
print('F1 Score:', f1score)
print('F1 Score micro:', f1score_global)
print('F1 Score macro:', f1score_individual)
print('F1 Score weighted:', f1score_weighted)

path = 'output/Simple Classifier/StatsWholeDataset.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/StatsWholeDataset.txt', 'w') as stats_output:
    

# TO DO:

# salvare csv direttamente? con tutti gli statistiche
# evaluation for -> trainset, dev set and dev set
# report results -> on an online document

'''