import json
import os
import re # regular expressions

from utils import write_output_stats_file, write_predictions_file, create_confusion_matrix

MAX_N_SENTENCES = 100

# SIMPLE CLASSIFIER FUNCTION
def simple_classifier(sents_ref_json):
    output_dict = []
    for sentence in sents_ref_json:
        label = [] 
        related = re.findall(r"data|information|cookies|account|content|\bad\b|website|advertiser", sentence['text'], re.IGNORECASE)
        violate = re.findall(r"collect|share|connect|we use|transfer|disclose|store|include|tailor|measure|we can|provide|automatically", sentence['text'], re.IGNORECASE) 
        commit = re.findall(r"settings|learn more|choose|manage|you can|delete|your consent|choices|protect|reduce|only|you have|restrict", sentence['text'], re.IGNORECASE) 
        opinion = re.findall(r"should|believe|we want|important|best|good", sentence['text'], re.IGNORECASE)

        if len(violate) >= 1:
            label.append('Violate privacy')
        if len(commit) >= 1 and len(commit) >= len(violate):
            label.insert(0, 'Commit to privacy')
        elif len(commit) >= 1:
            label.append('Commit to privacy')
        if len(opinion) >= 1:
            label.insert(0,'Declare opinion about privacy') # opinion related terms are stronger than commit and violate related terms
        if (len(related) >= 1 and label == []): # to use single label approach, add condition: or violate and commit
            label.append('Related to privacy') # related is only activated if none of the other categories were detected
        if label == []:
            label = ['Not applicable']

        output_dict.append({"text":sentence['text'], "label":label})

    return output_dict

#####
# MAIN

# Read input sentences
train_path = 'output/partition/multilabeldata_train.json'
dev_path = 'output/partition/multilabeldata_dev.json'
test_path = 'output/partition/multilabeldata_test.json'

with open(train_path) as file:
    document = file.read()
    train_sents_ref_json = json.loads(document)
    #print(train_sents_ref_json)
with open(dev_path) as file:
    document = file.read()
    dev_sents_ref_json = json.loads(document)
    #print(dev_sents_ref_json)
with open(test_path) as file:
    document = file.read()
    test_sents_ref_json = json.loads(document)
    #print(test_sents_ref_json)

# Getting list of multiple labels assigned to each sentence
train_labels_ref_list = [sent['label'] for sent in train_sents_ref_json]
dev_labels_ref_list = [sent['label'] for sent in dev_sents_ref_json]
test_labels_ref_list = [sent['label'] for sent in test_sents_ref_json]

# Filtering to list of primary labels
train_ref_primary_label = [label[0] for label in train_labels_ref_list]
dev_ref_primary_label = [label[0] for label in dev_labels_ref_list]
test_ref_primary_label = [label[0] for label in test_labels_ref_list]

# Feeding the classifier
train_pred_dict = simple_classifier(train_sents_ref_json)
dev_pred_dict = simple_classifier(dev_sents_ref_json)
test_pred_dict = simple_classifier(test_sents_ref_json)

# Output predictions in separate files
write_predictions_file(train_pred_dict, 'output/Simple Classifier/multilabelPredictions_Train.json')
write_predictions_file(dev_pred_dict,'output/Simple Classifier/multilabelPredictions_Dev.json')
write_predictions_file(test_pred_dict, 'output/Simple Classifier/multilabelPredictions_Test.json')

# Formatting predictions to calculate results
train_pred_array = [sent['label'] for sent in train_pred_dict]
train_pred_first_label = [label[0] for label in train_pred_array]

dev_pred_array = [sent['label'] for sent in dev_pred_dict]
dev_pred_first_label = [label[0] for label in dev_pred_array]

test_pred_array = [sent['label'] for sent in test_pred_dict]
test_pred_first_label = [label[0] for label in test_pred_array]

labels=['Commit to privacy', 'Declare opinion about privacy','Not applicable','Related to privacy','Violate privacy']
# Confusion Matrixes

path='output/Simple Classifier/1Label_confusion_matrix_TrainNormTrue.png'
create_confusion_matrix(train_ref_primary_label, train_pred_first_label, "true", path, labels, labels)
path='output/Simple Classifier/1Label_confusion_matrix_DevNormTrue.png'
create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, "true", path, labels, labels)
path='output/Simple Classifier/1Label_confusion_matrix_TestNormTrue.png'
create_confusion_matrix(test_ref_primary_label, test_pred_first_label, "true", path, labels, labels)
path='output/Simple Classifier/1Label_confusion_matrix_TrainNonNorm.png'
create_confusion_matrix(train_ref_primary_label, train_pred_first_label, None, path, labels, labels)
path='output/Simple Classifier/1Label_confusion_matrix_DevNonNorm.png'
create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, None, path, labels, labels)
path='output/Simple Classifier/1Label_confusion_matrix_TestNonNorm.png'
create_confusion_matrix(test_ref_primary_label, test_pred_first_label, None, path, labels, labels)

# Performance Measures

path='output/Simple Classifier/1labelPredictionsStats.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    print("Performance measures\n", file=file)
write_output_stats_file(path, "Train", train_ref_primary_label, train_pred_first_label, labels)
write_output_stats_file(path, "Dev", dev_ref_primary_label, dev_pred_first_label, labels)
write_output_stats_file(path, "Test", test_ref_primary_label, test_pred_first_label, labels)