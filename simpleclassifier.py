import json
import os
import re # regular expressions
from sklearn.metrics import precision_score, f1_score, ConfusionMatrixDisplay, recall_score, accuracy_score 
import matplotlib.pyplot as plt
from collections import Counter
import numpy

MAX_N_SENTENCES = 100

# FUNCTIONS

# SIMPLE CLASSIFIER
# TO DO: RECHOOSE RULES, CHOOSE ONLY WORDS WITH SEMANTIC MEANING
# remove the words that are not in the train set even though they make sense
# verbs - infinitive, noun root form
def simple_classifier(sents_ref_json):
    output_dict = []
    for sentence in sents_ref_json:
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
    return output_dict

# WRITE OUTPUT STATISTICS FILE
def write_output_stats_file(name, ref_labels, pred_labels):
    #path = 'output/Simple Classifier/1labelPredictionsStats_'+name+'.txt'
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    with open('output/Simple Classifier/1labelPredictionsStats_'+name+'.txt', 'a') as file:
        print("Performance in",name,"set:\n", file=file)
        print("Accuracy:",round( accuracy_score( ref_labels, pred_labels), 2), file=file)
        print("Precision micro:",round( precision_score( ref_labels, pred_labels, average="micro"), 2), file=file)
        print("Precision macro:",round( precision_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("Recall micro:",round( recall_score( ref_labels, pred_labels, average="micro"),2), file=file)
        print("Recall macro:",round( recall_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("F1 Score micro:",round( f1_score( ref_labels, pred_labels, average="micro"),2), file=file)
        print("F1 Score macro:",round( f1_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("F1 Score weighted:",round( f1_score(ref_labels, pred_labels, average="weighted"),2), file=file)

# WRITE OUTPUT PREDICTIONS IN JSON FORMAT
def write_predictions_file(name, pred_dict):
    path = 'output/Simple Classifier/multilabelPredictions_'+name+'.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open('output/Simple Classifier/multilabelPredictions_'+name+'.json', 'w') as file:
        file.write(json.dumps(pred_dict, indent=4, ensure_ascii=False))

# FLAG - THERE IS PROBABLY A PYTHON FUNCTION THAT CALCULATES THIS
def measure_distribution(ref_array, name):
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

    x_pos = numpy.arange(7) # sets number of bars
    plt.bar(x_pos, counter.values(),align='center')
    plt.xticks(x_pos, counter.keys(), rotation=45, ha="right") # sets labels of bars and their positions
    plt.subplots_adjust(bottom=0.4) # creates space for complete label names
    plt.savefig('output/Simple Classifier/multilabel_distribution_'+name+'.jpg')

    path = 'output/Simple Classifier/1labelPredictionsStats_'+name+'.txt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
   
    with open(path, 'w') as file:
        print(name+"set statistics:\n", file=file)
        print('Distribution of labels\n\nViolate privacy:', distr_violation,'\nCommit to privacy:', distr_commit,'\nOpinion about privacy:',distr_opinion, '\nRelated to privacy:', distr_related, '\nNot applicable:', distr_notApp,'\n', file=file)

def create_confusion_matrix(refs, preds,name):
    #ConfusionMatrixDisplay.from_predictions(ref_1label_str_list,pred_1label_str_list, normalize="true")
    ConfusionMatrixDisplay.from_predictions(refs,preds, normalize="true")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.4)
    #plt.show()
    plt.savefig('output/Simple Classifier/1Label_confusion_matrix_'+name+'.jpg')

#####
# MAIN

# Read input sentences
dev_path = 'output/partition/multilabeldata_dev.json'
train_path = 'output/partition/multilabeldata_train.json'
test_path = 'output/partition/multilabeldata_test.json'

# FLAG - CHECK IF RIGHT AND UPDATED FILE IS BEING PICKED 

with open(dev_path) as file:
    document = file.read()
    dev_sents_ref_json = json.loads(document)

dev_labels_ref_list = [sent['label'] for sent in dev_sents_ref_json]

# Count distribution + Multilabel distribution chart
measure_distribution(dev_labels_ref_list, "Dev")
# FLAG - CHECK IF DISTRIBUTION IS BEING MEASURED CORRECTLY

dev_pred_dict = simple_classifier(dev_sents_ref_json)

#write_predictions_file("Train", output_dict)
write_predictions_file("Dev", dev_pred_dict)
#write_predictions_file("Test", output_dict)
# FLAG - CHECK IF OUTPUT WAS CORRECTLY WRITTEN IN FILE

dev_pred_array = [sent['label'] for sent in dev_pred_dict]
dev_pred_first_label = [label[0] for label in dev_pred_array]
dev_ref_primary_label = [label[0] for label in dev_labels_ref_list]
# FLAG - CHECK IF PREDICTIONS WERE CORRECTLY FILTERED TO PRIMARY LABEL -- CHECKED

### OTHER APPROACHES FOR CHOOSING THE LABEL FOR EVALUATION -- check start of implementation at the end of the code

# CREATE DISTRIBUTION CHART OF ONLY 1 LABEL

# Multilabel distribution chart
#x_pos = numpy.arange(5) # sets number of bars
#plt.bar(x_pos, counter.values(),align='center')
#plt.xticks(x_pos, counter.keys(), rotation=45, ha="right") # sets labels of bars and their positions
#plt.subplots_adjust(bottom=0.4) # creates space for complete label names
#plt.savefig('output/Simple Classifier/First_label_distribution.jpg')
# FLAG - CHECK IF PREDICTIONS ARE CORRECTLY CALCULATED - ASK GABRIEL IF THERE IS AN AUTOMATED WAY TO DO IT

create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, "Dev")
# FLAG  - CHECK IF CONFUSION MATRIX IS CORRECT 

#labels = sorted(list(set(dev_ref_primary_label)))

#write_output_stats_file("Train", ref_primary_label, pred_first_label)
write_output_stats_file("Dev", dev_ref_primary_label, dev_pred_first_label)
#write_output_stats_file("Test", ref_primary_label, pred_first_label)



###############
# TO DO:

# salvare csv direttamente? con tutti gli statistiche
# evaluation for -> trainset, dev set and dev set
# report results -> on an online document


# APPENDIX
'''
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