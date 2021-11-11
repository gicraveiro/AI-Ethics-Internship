import json
import os
import re # regular expressions
from sklearn.metrics import precision_score, confusion_matrix, f1_score
#from sklearn.metrics import a

# Read input sentences
path = 'output/partition/fbdata_complete.json'

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences_ref = json.loads(document)

ref_array = [sent['label'] for sent in json_sentences_ref]

# DEBUG PRINTS
#for sentence in json_sentences:
#    print(sentence)
#    print(sentence['text'])
#    print(sentence['label'],'\n\n')
output_dict = []
label = ''
# SIMPLE CLASSIFIER
for sentence in json_sentences_ref:
    label = 'Not applicable'
    related = re.search(r".*data.*|.*information.*|.*cookies.*|.*personalize.*|.*content.*", sentence['text'], re.IGNORECASE)
    violate = re.search(r".*collect.*|.*share.*|.*connect.*|.*off facebook.*|.*receive information about you.*|.*\bsee\b.*", sentence['text'], re.IGNORECASE) # |.*see.*
    commit = re.search(r".*[instagram|facebook] settings.*|.*learn.*|.*choose.*|.*control.*|.*manage.*|.*opt out.*|.*delete.*|.*your consent.*|.*allow.*|.*change.*|.*choices.*|.*select.*|.*require.*|.*protection.*|.*reduce.*|.*don't sell.*", sentence['text'], re.IGNORECASE) # if 1/2 if you choose, protection(YES)law?(NO), you will have control, report(no...), will not be able, restrict, reduce, remove, don't sell, impose []restrictions, permission, you [can/will have] control, preferences(not ueful in this case), you own
    opinion = re.search(r".*should.*|.*believe.*", sentence['text'], re.IGNORECASE)

    if violate:
        label = 'Violate privacy'
    if commit:
        label = 'Commit to privacy'
    if (related and label == 'Not applicable') or (violate and commit):
        label = 'Related to privacy'
    if opinion:
        label = 'Declare opinion about privacy'

    output_dict.append({"text":sentence['text'], "label":label})

#print(output_dict)

# WRITE OUTPUT
path = 'output/Simple Classifier/fbdata_complete.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/fbdata_complete.json', 'w') as train_file:
    train_file.write(json.dumps(output_dict, indent=4, ensure_ascii=False))

with open(path) as file:
    document = file.read()
    json_sentences_predicted = json.loads(document)

pred_array = [sent['label'] for sent in json_sentences_predicted]

confusion_matr = confusion_matrix(ref_array,pred_array)
precision = precision_score(ref_array, pred_array, average=None)
f1score_global = f1_score(ref_array, pred_array, average='micro')
f1score_individual = f1_score(ref_array, pred_array, average='macro')
f1score_weighted = f1_score(ref_array, pred_array, average='weighted')

print(confusion_matr)
print(precision)
print(f1score_global)
print(f1score_individual)
print(f1score_weighted)
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