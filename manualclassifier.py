import json
import os
import re # regular expressions

# Read input sentences
path = 'output/partition/fbdata_train.json'

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences_ref = json.loads(document)

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
    related = re.search(".*privacy.*", sentence['text'], re.IGNORECASE)
    violate = re.search(".*collect.*", sentence['text'], re.IGNORECASE)
    commit = re.search(".*settings.*|.*learn.*|.*choose.*|.*preferences.*|.*control.*|.*manage.*|.*opt out.*|.*delete.*|.*consent.*|.*allow.*|.*change.*|.*choices.*", sentence['text'], re.IGNORECASE)
    opinion = re.search(".*should.*|.*believe.*", sentence['text'], re.IGNORECASE)

    if violate:
        label = 'Violate privacy'
    if commit:
        label = 'Commit to privacy'
    if opinion:
        label = 'Opinion about privacy'
    if (related and label == 'Not applicable') or (violate and commit):
        label = 'Related to privacy'
        #if violate and commit:
        #    print("VIOLATE AND COMMIT")
        
    output_dict.append({"text":sentence['text'], "label":label})

#print(output_dict)

# WRITE OUTPUT
path = 'output/Simple Classifier/fbdata_train.json'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/Simple Classifier/fbdata_train.json', 'w') as train_file:
    train_file.write(json.dumps(output_dict, indent=4, ensure_ascii=False))

with open(path) as file:
    document = file.read()
    json_sentences_result = json.loads(document)

total_tp = 0
violate_tp = 0
commit_tp = 0
opinion_tp = 0
related_tp = 0
nonAp_tp = 0
for sent_ref, sent_result in zip(json_sentences_ref, json_sentences_result):
    if sent_ref['label'] == sent_result['label']:
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

print('Results statistics')
print('Recall\n\nViolate privacy\n',violate_tp,'/79\n\n','Commit to privacy\n',commit_tp,'/139\n\n','Opinion about privacy\n',opinion_tp,'/14\n\n','Related to privacy\n',related_tp,'/128\n\n','Not applicable\n',nonAp_tp,'/338')
#print('Precision\n\nViolate privacy\n',violate_tp,'/79\n\n','Commit to privacy\n',commit_tp,'/139\n\n','Opinion about privacy\n',opinion_tp,'/14\n\n','Related to privacy\n',related_tp,'/128\n\n','Not applicable\n',nonAp_tp,'/338')