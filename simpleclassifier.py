import json
import os
import re # regular expressions
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import Counter
import numpy
from utils import write_output_stats_file

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


# WRITE OUTPUT PREDICTIONS IN JSON FORMAT
def write_predictions_file(name, pred_dict):
    path = 'output/Simple Classifier/multilabelPredictions_'+name+'.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open('output/Simple Classifier/multilabelPredictions_'+name+'.json', 'w') as file:
        file.write(json.dumps(pred_dict, indent=4, ensure_ascii=False))

'''
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

    plt.clf() # cleans previous graph
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
'''

def create_confusion_matrix(refs, preds,name):
    #ConfusionMatrixDisplay.from_predictions(ref_1label_str_list,pred_1label_str_list, normalize="true")
    ConfusionMatrixDisplay.from_predictions(refs,preds, normalize="true")
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.4)
    #plt.show()
    plt.savefig('output/Simple Classifier/1Label_confusion_matrix_'+name+'.jpg')

# CREATE DISTRIBUTION CHART OF ONLY 1 LABEL
    # Multilabel distribution chart
def plot_distribution(counter, name, type):    
    plt.clf() # cleans previous graphs
    x_pos = numpy.arange(len(counter)) # sets number of bars
    plt.bar(x_pos, counter.values(),align='center')
    plt.xticks(x_pos, counter.keys(), rotation=45, ha="right") # sets labels of bars and their positions
    plt.subplots_adjust(bottom=0.45, left=0.25) # creates space for complete label names
    for i, value in enumerate(counter.values()):
        plt.text(i,value,str(value))
    plt.ylim((0,counter.most_common()[0][1]+counter.most_common()[2][1]))
    #plt.show()
    plt.savefig('output/Simple Classifier/'+type+'_distribution_'+name+'.jpg')
    return 

def calculate_distribution(label_count, total):
    return round(label_count/total, 2)

def write_distribution(path,counter):
    total = sum(counter.values())

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        print('Distribution of labels\n', file=file)
        for item in counter.items():
            print(item[0], calculate_distribution(item[1], total), file=file)
        print('\n',file=file)

    
#####
# MAIN

# Read input sentences
train_path = 'output/partition/multilabeldata_train.json'
dev_path = 'output/partition/multilabeldata_dev.json'
test_path = 'output/partition/multilabeldata_test.json'

# FLAG - CHECK IF RIGHT AND UPDATED FILE IS BEING PICKED 

with open(train_path) as file:
    document = file.read()
    train_sents_ref_json = json.loads(document)
with open(dev_path) as file:
    document = file.read()
    dev_sents_ref_json = json.loads(document)
with open(test_path) as file:
    document = file.read()
    test_sents_ref_json = json.loads(document)

train_labels_ref_list = [sent['label'] for sent in train_sents_ref_json]
dev_labels_ref_list = [sent['label'] for sent in dev_sents_ref_json]
#test_labels_ref_list = [sent['label'] for sent in test_sents_ref_json]

# Multilabel distribution chart
counter = Counter(tuple(item) for item in train_labels_ref_list)
plot_distribution(counter, "Train", "multilabel")
write_distribution('output/Simple Classifier/multilabelPredictionsStats_Train.txt', counter)
counter = Counter(tuple(item) for item in dev_labels_ref_list)
plot_distribution(counter, "Dev", "multilabel")
write_distribution('output/Simple Classifier/multilabelPredictionsStats_Dev.txt', counter)
#counter = Counter(tuple(item) for item in test_labels_ref_list)
#plot_distribution(counter, "Test", "multilabel")
#write_distribution('output/Simple Classifier/multilabelPredictionsStats_Test.txt', counter)

# Count distribution ---- PAUSED FOR NOW
#measure_distribution(train_labels_ref_list, "Train")
#measure_distribution(dev_labels_ref_list, "Dev")
#measure_distribution(test_labels_ref_list, "Test")
# FLAG - CHECK IF DISTRIBUTION IS BEING MEASURED CORRECTLY

train_pred_dict = simple_classifier(train_sents_ref_json)
dev_pred_dict = simple_classifier(dev_sents_ref_json)
#test_pred_dict = simple_classifier(test_sents_ref_json)

write_predictions_file("Train", train_pred_dict)
write_predictions_file("Dev", dev_pred_dict)
#write_predictions_file("Test", test_pred_dict)
# FLAG - CHECK IF predictions Were CORRECTLY WRITTEN IN FILE

train_pred_array = [sent['label'] for sent in train_pred_dict]
train_pred_first_label = [label[0] for label in train_pred_array]
train_ref_primary_label = [label[0] for label in train_labels_ref_list]

dev_pred_array = [sent['label'] for sent in dev_pred_dict]
dev_pred_first_label = [label[0] for label in dev_pred_array]
dev_ref_primary_label = [label[0] for label in dev_labels_ref_list]

#test_pred_array = [sent['label'] for sent in test_pred_dict]
#test_pred_first_label = [label[0] for label in test_pred_array]
#test_ref_primary_label = [label[0] for label in test_labels_ref_list]

# FLAG - CHECK IF PREDICTIONS WERE CORRECTLY FILTERED TO PRIMARY LABEL -- CHECKED

### OTHER APPROACHES FOR CHOOSING THE LABEL FOR EVALUATION -- check start of implementation at the end of the code
counter = Counter(train_ref_primary_label)
plot_distribution(counter, "Train", "First_label")
write_distribution('output/Simple Classifier/1labelPredictionsStats_Train.txt', counter)
counter = Counter(dev_ref_primary_label)
plot_distribution(counter,"Dev", "First_label")
write_distribution('output/Simple Classifier/1labelPredictionsStats_Dev.txt', counter)
#counter = Counter(test_ref_primary_label)
#plot_distribution(counter,"Test")
#write_distribution('output/Simple Classifier/1labelPredictionsStats_Test.txt', counter)

# FLAG - CHECK IF PREDICTIONS ARE CORRECTLY CALCULATED - ASK GABRIEL IF THERE IS AN AUTOMATED WAY TO DO IT

create_confusion_matrix(train_ref_primary_label, train_pred_first_label, "Train")
create_confusion_matrix(dev_ref_primary_label, dev_pred_first_label, "Dev")
#create_confusion_matrix(test_ref_primary_label, test_pred_first_label, "Dev")

# FLAG  - CHECK IF CONFUSION MATRIX IS CORRECT 

write_output_stats_file('output/Simple Classifier/1labelPredictionsStats_Train.txt', "Train", train_ref_primary_label, train_pred_first_label)
write_output_stats_file('output/Simple Classifier/1labelPredictionsStats_Dev.txt', "Dev", dev_ref_primary_label, dev_pred_first_label)
#write_output_stats_file('output/Simple Classifier/1labelPredictionsStats_Test.txt', "Test", test_ref_primary_label, test_pred_first_label)

# FLAG - CHECK IF STATS WERE CALCULATED AND WRITTEN CORRECTLY


###############
# TO DO:

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