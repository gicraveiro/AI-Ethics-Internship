import json
import os
import re # regular expressions
from sklearn.metrics import precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay, recall_score
import matplotlib.pyplot as plt
from collections import Counter
import csv
from googleapiclient.discovery import build
from google.oauth2 import service_account
import numpy
#from sklearn.metrics import a

# Read input sentences
#path = 'output/partition/fbdata_test.json'
path = 'output/partition/multilabeldata_test.json'

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences_ref = json.loads(document)

ref_array = [sent['label'] for sent in json_sentences_ref]
counter = Counter(tuple(item) for item in ref_array)
tot_sents = len(ref_array)
distr_violation = round (ref_array.count(['Violate privacy'])/tot_sents, 2)
distr_commit = round( ref_array.count(['Commit to privacy'])/tot_sents, 2)
distr_related = round (ref_array.count(['Related to privacy'])/tot_sents, 2)
distr_opinion = round(ref_array.count(['Declare opinion about privacy'])/tot_sents, 2)
distr_notApp = round(ref_array.count(['Not applicable'])/tot_sents, 2)
count_violation = ref_array.count(['Violate privacy'])
count_commit = ref_array.count(['Commit to privacy'])
count_related = ref_array.count(['Related to privacy'])
count_opinion = ref_array.count(['Declare opinion about privacy'])
count_notApp = ref_array.count(['Not applicable'])

x_pos = numpy.arange(7) # sets number of bars
plt.bar(x_pos, counter.values(),align='center')
plt.xticks(x_pos, counter.keys(), rotation=45, ha="right") # sets labels of bars and their positions
plt.subplots_adjust(bottom=0.4) # creates space for complete label names
plt.savefig('output/Simple Classifier/multilabel_distribution.jpg')

print('Distribution of labels\nViolate privacy:', distr_violation,'\nCommit to privacy:', distr_commit,'\nOpinion about privacy:',distr_opinion, '\nRelated to privacy:', distr_related, '\nNot applicable:', distr_notApp,'\n')

#print('Distribution of labels\nViolate privacy:', count_violation,'\nCommit to privacy:', count_commit,'\nOpinion about privacy:',count_opinion, '\nRelated to privacy:', count_related, '\nNot applicable:', count_notApp,'\n')

# DEBUG PRINTS
#for sentence in json_sentences:
#    print(sentence)
#    print(sentence['text'])
#    print(sentence['label'],'\n\n')
output_dict = []
#label = [] # = ''
# SIMPLE CLASSIFIER
for sentence in json_sentences_ref:
    label = [] # label = 'Not applicable'
    related = re.search(r".*data.*|.*information.*|.*cookies.*|.*personalize.*|.*content.*", sentence['text'], re.IGNORECASE)
    violate = re.search(r".*collect.*|.*share.*|.*connect.*|.*off facebook.*|.*receive information about you.*|.*\bsee\b.*", sentence['text'], re.IGNORECASE) # |.*see.*
    commit = re.search(r".*[instagram|facebook] settings.*|.*learn.*|.*choose.*|.*control.*|.*manage.*|.*opt out.*|.*delete.*|.*your consent.*|.*allow.*|.*change.*|.*choices.*|.*select.*|.*require.*|.*protection.*|.*reduce.*|.*don't sell.*", sentence['text'], re.IGNORECASE) # if 1/2 if you choose, protection(YES)law?(NO), you will have control, report(no...), will not be able, restrict, reduce, remove, don't sell, impose []restrictions, permission, you [can/will have] control, preferences(not ueful in this case), you own
    opinion = re.search(r".*should.*|.*believe.*", sentence['text'], re.IGNORECASE)

# TO DO: add weight to decide primary and secondary labels
    if violate:
        #label = 'Violate privacy'
        label.append('Violate privacy')
    if commit:
        label = 'Commit to privacy'
        label.append('Commit to privacy')
    if opinion:
        label.insert(0,'Declare opinion about privacy') # opinion related terms are stronger than commit and violate related terms
    if (related and label == []): # for single label rule, add condition: or violate and commit
        label.append('Related to privacy') # related is only activated if none of the other categories were detected
    if label == []:
        label = ['Not applicable']

    output_dict.append({"text":sentence['text'], "label":label})

#print(output_dict)

# WRITE OUTPUT
path = 'output/Simple Classifier/multilabeldata_test.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/multilabeldata_test.json', 'w') as train_file:
    train_file.write(json.dumps(output_dict, indent=4, ensure_ascii=False))

with open(path) as file:
    document = file.read()
    json_sentences_predicted = json.loads(document)

pred_array = [sent['label'] for sent in json_sentences_predicted]

#confusion_matr = confusion_matrix(ref_array,pred_array, normalize="true")
ConfusionMatrixDisplay.from_predictions(ref_array,pred_array, normalize="true")
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.4)
#plt.show()
plt.savefig('output/Simple Classifier/multilabel_confusion_matrix.jpg')
labels = sorted(list(set(ref_array)))
#print(labels)
precision = precision_score(ref_array, pred_array,labels=labels, average=None) # 
precision_micro = precision_score(ref_array, pred_array, average='micro')
#precision_macro = precision_score(ref_array, pred_array, average='macro')
recall = recall_score(ref_array, pred_array, average=None)
recall_micro = recall_score(ref_array, pred_array, average='micro')
f1score = f1_score(ref_array, pred_array, average=None)
f1score_global = f1_score(ref_array, pred_array, average='micro')
f1score_individual = f1_score(ref_array, pred_array, average='macro')
#f1score_weighted = f1_score(ref_array, pred_array, average='weighted')

precision = precision.tolist()
recall = recall.tolist()
f1score = f1score.tolist()
#print(type(labels))
#print(precision.tolist())

with open('output/Simple Classifier/SimpleClassifierMultiLabelStats.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    labels.insert(0, 'Classes')
    precision.insert(0, 'Precision')
    recall.insert(0, 'Recall')
    f1score.insert(0, 'F1 Score')
    #print(type(labels), labels)
    filewriter.writerow(labels)
    filewriter.writerow(precision)
    filewriter.writerow(recall)
    filewriter.writerow(f1score)

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

values = [] # list, TO DO: ADD STUFF IN IT? CHANGE TO CSV APPROACH OF WRITING MAYBE?
values.append(precision)
values.append(recall)
values.apo
sentences = {
        'values': values
    }

sheet.values().update(spreadsheetId=SAMPLE_SPREADSHEET_ID, range='Test'+'!A2:F500',valueInputOption=value_input_option, body=sentences).execute()


#print('Confusion matrix:\n',confusion_matr)
print()
print('Precision:',precision)
print('Precision:',precision_micro)
print('Recall:',recall)
print('Recall micro:',recall_micro)
print('F1 Score:', f1score)
print('F1 Score micro:', f1score_global)
print('F1 Score macro:', f1score_individual)
#print('F1 Score weighted:', f1score_weighted)


# TO DO:

# MAKE LABELS READABLE - DONE
# salvare csv direttamente? con tutti gli statistiche
# CHANGE TO TEST SET!! - DONE


'''
total_tp = 0
violate_tp = 0
commit_tp = 0
opinion_tp = 0
related_tp = 0
nonAp_tp = 0
total_fp = 0
violate_fp = 0
commit_fp = 0
opinion_fp = 0
related_fp = 0
nonAp_fp = 0
total_fn = 0
violate_fn = 0
commit_fn = 0
opinion_fn = 0
related_fn = 0
nonAp_fn = 0

for sent_ref, sent_predicted in zip(json_sentences_ref, json_sentences_predicted):
    if sent_ref['label'] == sent_predicted['label']:
        total_tp += 1
        if sent_ref['label'] == 'Violate privacy':
            violate_tp += 1
        elif sent_ref['label'] == 'Commit to privacy':
            commit_tp += 1
        elif sent_ref['label'] == 'Opinion about privacy':
            opinion_tp += 1
        elif sent_ref['label'] == 'Related to privacy':
            related_tp += 1
        else:
            nonAp_tp += 1
    else:
        total_fp += 1
        total_fn += 1
        if sent_ref['label'] == 'Violate privacy':
            violate_fn += 1
        elif sent_ref['label'] == 'Commit to privacy':
            commit_fn += 1
        elif sent_ref['label'] == 'Declare opinion about privacy':
            opinion_fn += 1
        elif sent_ref['label'] == 'Related to privacy':
            related_fn += 1
        else:
            nonAp_fn += 1
        
        if sent_predicted['label'] == 'Violate privacy':
            violate_fp += 1
        elif sent_predicted['label'] == 'Commit to privacy':
            commit_fp += 1
        elif sent_predicted['label'] == 'Declare opinion about privacy':
            opinion_fp += 1
        elif sent_predicted['label'] == 'Related to privacy':
            related_fp += 1
        else:
            nonAp_fp += 1


path = 'output/Simple Classifier/StatsWholeDataset.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/StatsWholeDataset.txt', 'w') as stats_output:
    print('Resulted statistics\n', file=stats_output)
#print('Recall\n\nViolate privacy\n',violate_tp,'/79\n','Commit to privacy\n',commit_tp,'/139\n','Opinion about privacy\n',opinion_tp,'/14\n','Related to privacy\n',related_tp,'/128\n','Not applicable\n',nonAp_tp,'/338\n','Total\n',total_tp,'/698\n')
    print('Recall\n\nViolate privacy\n',violate_tp,'/',violate_fn+violate_tp,'\n','Commit to privacy\n',commit_tp,'/',commit_fn+commit_tp,'\n','Opinion about privacy\n',opinion_tp,'/',opinion_fn+opinion_tp,'\n','Related to privacy\n',related_tp,'/',related_fn+related_tp,'\n','Not applicable\n',nonAp_tp,'/',nonAp_fn+nonAp_tp,'\nTotal\n',total_tp,'/',total_fn+total_tp,'\n', file=stats_output)
    print('Precision\n\nViolate privacy\n',violate_tp,'/',violate_fp+violate_tp,'\n','Commit to privacy\n',commit_tp,'/',commit_fp+commit_tp,'\n','Opinion about privacy\n',opinion_tp,'/',opinion_fp+opinion_tp,'\n','Related to privacy\n',related_tp,'/',related_fp+related_tp,'\n','Not applicable\n',nonAp_tp,'/',nonAp_fp+nonAp_tp,'\nTotal\n',total_tp,'/',total_fp+total_tp, file=stats_output)
    '''

    ## ADABOOST  CLASSIFIER SKLEARN LIBRARY