import json
import os
import re # regular expressions

from utils import write_output_stats_file, write_predictions_file, create_confusion_matrix

MAX_N_SENTENCES = 100

# FUNCTIONS

# SIMPLE CLASSIFIER
def simple_classifier(sents_ref_json):
    output_dict = []
    for sentence in sents_ref_json:
        label = [] # label = 'Not applicable'
        # OLD RULES (the ones I reported in the monografia)
        #related = re.search(r".*data.*|.*information.*|.*cookies.*|.*personalize.*|.*content.*", sentence['text'], re.IGNORECASE)
        #violate = re.search(r".*collect.*|.*share.*|.*connect.*|.*off facebook.*|.*receive information about you.*|.*\bsee\b.*", sentence['text'], re.IGNORECASE) # |.*see.*
        #commit = re.search(r".*[instagram|facebook] settings.*|.*learn.*|.*choose.*|.*control.*|.*manage.*|.*opt out.*|.*delete.*|.*your consent.*|.*allow.*|.*change.*|.*choices.*|.*select.*|.*require.*|.*protection.*|.*reduce.*|.*don't sell.*", sentence['text'], re.IGNORECASE) # if 1/2 if you choose, protection(YES)law?(NO), you will have control, report(no...), will not be able, restrict, reduce, remove, don't sell, impose []restrictions, permission, you [can/will have] control, preferences(not ueful in this case), you own
        #opinion = re.search(r".*should.*|.*believe.*", sentence['text'], re.IGNORECASE)
        # NEW RULES (analyzed only in training set and with semantic meaning in max 2 terms)
        related = re.findall(r"data|information|cookies|account|content|\bad\b|website|advertiser", sentence['text'], re.IGNORECASE)
        violate = re.findall(r"collect|share|connect|we use|transfer|disclose|store|include|tailor|measure|we can|provide|automatically", sentence['text'], re.IGNORECASE) 
        commit = re.findall(r"settings|learn more|choose|manage|you can|delete|your consent|choices|protect|reduce|only|you have|restrict", sentence['text'], re.IGNORECASE) 
        opinion = re.findall(r"should|believe|we want|important|best|good", sentence['text'], re.IGNORECASE)

    # FLAG - CHECK IF RULES DO WHAT I INTENDED - not every single word but checked some examples

    # TO DO: add weight to decide primary and secondary labels DONE
        if len(violate) >= 1:
            label.append('Violate privacy')
        if len(commit) >= 1 and len(commit) >= len(violate):
            label.insert(0, 'Commit to privacy')
        elif len(commit) >= 1:
            label.append('Commit to privacy')
        if len(opinion) >= 1:
            label.insert(0,'Declare opinion about privacy') # opinion related terms are stronger than commit and violate related terms
        if (len(related) >= 1 and label == []): # for single label rule, add condition: or violate and commit
            label.append('Related to privacy') # related is only activated if none of the other categories were detected
        if label == []:
            label = ['Not applicable']

        output_dict.append({"text":sentence['text'], "label":label})
        # DEBUG PRINTS
        #print("\n",{"text":sentence['text'], "label":label}, "\n", "violate", violate, len(violate), "\n", "commit", commit, len(commit), "\n", "opinion", opinion, len(opinion), "\n", "related", related, len(related) )
        #print(output_dict, {"text":sentence['text'], "label":label})
    return output_dict


#####
# MAIN

# Read input sentences
train_path = 'output/partition/multilabeldata_train.json'
dev_path = 'output/partition/multilabeldata_dev.json'
test_path = 'output/partition/multilabeldata_test.json'

# FLAG - CHECK IF RIGHT AND UPDATED FILE IS BEING PICKED - CHECKED

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

train_labels_ref_list = [sent['label'] for sent in train_sents_ref_json]
dev_labels_ref_list = [sent['label'] for sent in dev_sents_ref_json]
test_labels_ref_list = [sent['label'] for sent in test_sents_ref_json]

# FLAG - CHECKED

# Single label distribution count + chart
train_ref_primary_label = [label[0] for label in train_labels_ref_list]
dev_ref_primary_label = [label[0] for label in dev_labels_ref_list]
test_ref_primary_label = [label[0] for label in test_labels_ref_list]

# FLAG - CHECKED

train_pred_dict = simple_classifier(train_sents_ref_json)
dev_pred_dict = simple_classifier(dev_sents_ref_json)
test_pred_dict = simple_classifier(test_sents_ref_json)

# FLAG - CHECKED

write_predictions_file("Train", train_pred_dict)
write_predictions_file("Dev", dev_pred_dict)
write_predictions_file("Test", test_pred_dict)

# FLAG - CHECK IF predictions Were CORRECTLY WRITTEN IN FILE - CHECKED

train_pred_array = [sent['label'] for sent in train_pred_dict]
train_pred_first_label = [label[0] for label in train_pred_array]

dev_pred_array = [sent['label'] for sent in dev_pred_dict]
dev_pred_first_label = [label[0] for label in dev_pred_array]

test_pred_array = [sent['label'] for sent in test_pred_dict]
test_pred_first_label = [label[0] for label in test_pred_array]

# FLAG - CHECK IF PREDICTIONS WERE CORRECTLY FILTERED TO PRIMARY LABEL and aligned with the ones before -- CHECKED

path='output/Simple Classifier/1Label_confusion_matrix_TrainNormTrue.png'
create_confusion_matrix(train_ref_primary_label, train_pred_first_label, "true", path, None, None)
path='output/Simple Classifier/1Label_confusion_matrix_DevNormTrue.png'
create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, "true", path, None, None)
path='output/Simple Classifier/1Label_confusion_matrix_TestNormTrue.png'
create_confusion_matrix(test_ref_primary_label, test_pred_first_label, "true", path, None, None)
path='output/Simple Classifier/1Label_confusion_matrix_TrainNonNorm.png'
create_confusion_matrix(train_ref_primary_label, train_pred_first_label, None, path, None, None)
path='output/Simple Classifier/1Label_confusion_matrix_DevNonNorm.png'
create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, None, path, None, None)
path='output/Simple Classifier/1Label_confusion_matrix_TestNonNorm.png'
create_confusion_matrix(test_ref_primary_label, test_pred_first_label, None, path, None, None)

# FLAG  - CHECK IF CONFUSION MATRIX IS CORRECT - CHECKED

path='output/Simple Classifier/1labelPredictionsStats.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    print("Performance measures\n", file=file)
write_output_stats_file(path, "Train", train_ref_primary_label, train_pred_first_label)
write_output_stats_file(path, "Dev", dev_ref_primary_label, dev_pred_first_label)
write_output_stats_file(path, "Test", test_ref_primary_label, test_pred_first_label)

# FLAG - CHECK IF STATS WERE CALCULATED AND WRITTEN CORRECTLY - checked

'''
###############

# APPENDIX
#
# SKETCH OF OTHER APPROACHES TO SELECT LABELS (CONSIDERING MULTILABEL OR CHOOSING PRIMARY LABEL DIFFERENTLY)

# Multilabel distribution count + chart
counter = Counter(tuple(item) for item in train_labels_ref_list)
plot_distribution(counter, "Train", "multilabel")
write_distribution('output/Simple Classifier/multilabelPredictionsStats_Train.txt', counter)
counter = Counter(tuple(item) for item in dev_labels_ref_list)
plot_distribution(counter, "Dev", "multilabel")
write_distribution('output/Simple Classifier/multilabelPredictionsStats_Dev.txt', counter)
#counter = Counter(tuple(item) for item in test_labels_ref_list)
#plot_distribution(counter, "Test", "multilabel")
#write_distribution('output/Simple Classifier/multilabelPredictionsStats_Test.txt', counter)

#####

counter = Counter(train_ref_primary_label)
plot_distribution(counter, "Train", "First_label")
write_distribution('output/Simple Classifier/1labelPredictionsStats_Train.txt', counter)
counter = Counter(dev_ref_primary_label)
plot_distribution(counter,"Dev", "First_label")
write_distribution('output/Simple Classifier/1labelPredictionsStats_Dev.txt', counter)
#counter = Counter(test_ref_primary_label)
#plot_distribution(counter,"Test")
#write_distribution('output/Simple Classifier/1labelPredictionsStats_Test.txt', counter)

#######

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
'''
