# Divides annotated dataset in train set, dev set and test set
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy
import spacy
from utils import clean_corpus, reconstruct_hyphenated_words

# Functions

# Creates dictionary of a set, associating sentence with label
def create_sent_label_dict(sents, labels):
    sents_dict = []
    for row_id,row in enumerate(sents):
        row = re.sub("\n", " ", row)
        sents_dict.append({"text":row.strip(), "label":labels[row_id]})
    return sents_dict

# Writes json of partition set, each entry is the sentence associated with its labels
def write_partition_file(partition_dict, name):
    path = 'output/partition/multilabeldata_'+name+'.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open('output/partition/multilabeldata_'+name+'.json', 'w') as file:
        file.write(json.dumps(partition_dict, indent=4, ensure_ascii=False))

# creating list of labels and count its distribution
def create_label_list(name):
    with open('output/partition/multilabeldata_'+name+'.json', 'r') as file:
        json_obj = json.loads(file.read())
        count = []
        for i in json_obj:
            item = i['label']
            item = str(item)
            count.append(item)
        print("\n",Counter(count), "\n")
    return count

# Prints distribution of labels in one partition
def print_partition_distribution(name, count, sum):
    print("\n",name," set")
    for item, sum_item in zip(sorted(Counter(count)), sorted(Counter(sum))):
        distr_value = Counter(count)[item]
        total_value = Counter(sum)[sum_item]
        print(item,  sum_item)
        print(distr_value, "out of", total_value, "samples in the whole dataset")
        print(round(float(distr_value)/float(total_value)*100,2),'\n')

def plot_distribution(counter, name, type):    
    total = sum(counter.values())
    counter = counter.most_common()
    values = [float(item[1]) for item in counter]
    keys = [str(item[0]) for item in counter]
    plt.clf() # cleans previous graphs
    x_pos = numpy.arange(len(counter)) # sets number of bars
    plt.bar(x_pos, values,align='center')
    plt.xticks(x_pos, keys, rotation=45, ha="right") # sets labels of bars and their positions
    plt.subplots_adjust(bottom=0.5, left=0.2) # creates space for complete label names
    # TO DO: adding a title might be a good idea -> plt.title('Distribution ...')
    plt.xlabel('Labels',labelpad=10.0)
    plt.ylabel('Frequency (%)',labelpad=10.0)
    for i, item in enumerate(values):
        plt.text(i,item,str(round((item*100/total),1)))
    plt.ylim((0,values[0]+values[2]))
    #plt.show()
    plt.savefig('output/partition/'+type+'_distribution_'+name+'.png')
    return 

def calculate_distribution(label_count, total):
    return round(label_count/total, 3)

def write_distribution(path,counter,name):
    total = sum(counter.values())
    with open(path, 'a') as file:
        print(name,"set\n", file=file)
        for item in counter.items():
            print(item[0], calculate_distribution(item[1], total), " ("+str(item[1])+"/"+str(total)+")", file=file)
        print("\n",file=file)
    
def remove_empty_sentences(sents, labels):
    for i, (sent, label) in enumerate(zip(sents, labels)):
        cleared_sent = clean_corpus(sent)
        cleared_sent = nlp(cleared_sent)
        cleared_sent = reconstruct_hyphenated_words(cleared_sent)
        cleared_sent = [token.text for token in cleared_sent if not token.is_space if not token.is_punct]
        # OBS: MAYBE ENHANCING PREPROCESSING BY REMOVING LITTLE SQUARES COULD BE AN OPTION - but important to change it in the partition file too
        if (label == ['Not applicable'] and len(cleared_sent) == 0):
            sents[i] = "REMOVE THIS ITEM"
            labels[i] = "REMOVE THIS ITEM"
    sents = [sent for sent in sents if sent != "REMOVE THIS ITEM"]
    labels = [label for label in labels if label != "REMOVE THIS ITEM"]
    return sents, labels
# MAIN

# Reads annotation table from file .csv saved locally and creates labels and senences list
annotation = pd.read_csv("data/Facebook/Annotated/AnnotatedMultiLabelDataset.csv")
sents = annotation['Sentences'].values
labels1 = annotation['Primary Label'].values
labels2 = annotation['Secondary Label'].values

# Formatting labels (because unfilled secondary labels must be treated accordingly)
labels = []
for l1,l2 in zip(labels1,labels2):
    row_labels = []
    row_labels.append(l1)
    if (type(l2) == str):
        row_labels.append(l2)
    labels.append(row_labels)

# Partitions data into 80% trainset and remaining 20%
sents_train, sents_test, labels_train, labels_test = train_test_split(sents,labels, test_size=0.2, stratify=labels, random_state=1111111)

# Partitions remaining 20% into dev set (10%) and test set (10%)
sents_test, sents_dev, labels_test, labels_dev = train_test_split(sents_test,labels_test, test_size=0.5, stratify=labels_test, random_state=1111111)

nlp = spacy.load('en_core_web_lg',disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']) # this was to save memory

# Preprocessing
# Removes empty sentences
sents_train, labels_train = remove_empty_sentences(sents_train, labels_train)
sents_dev, labels_dev = remove_empty_sentences(sents_dev, labels_dev)
sents_test, labels_test = remove_empty_sentences(sents_test, labels_test)
# creates a dictionary with sentences and their labels
train_dict = create_sent_label_dict(sents_train, labels_train)
dev_dict = create_sent_label_dict(sents_dev, labels_dev)
test_dict = create_sent_label_dict(sents_test, labels_test)
total_dict = train_dict + dev_dict + test_dict

# creates json files with reference sentences and their labels as output
write_partition_file(train_dict, 'train')
write_partition_file(dev_dict, 'dev')
write_partition_file(test_dict, 'test')

# Get reference labels list
total_labels_ref_list = [sent['label'] for sent in total_dict]
train_labels_ref_list = [sent['label'] for sent in train_dict]
dev_labels_ref_list = [sent['label'] for sent in dev_dict]
test_labels_ref_list = [sent['label'] for sent in test_dict]

# Multilabel distribution counts + graphs
path= 'output/partition/multilabelDistribution.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    print("Distribution of labels\n", file=file)
counter = Counter(tuple(item) for item in total_labels_ref_list)
plot_distribution(counter, "Total", "multilabel")
write_distribution(path, counter, "Total")
counter = Counter(tuple(item) for item in train_labels_ref_list)
plot_distribution(counter, "Train", "multilabel")
write_distribution(path, counter, "Train")
counter = Counter(tuple(item) for item in dev_labels_ref_list)
plot_distribution(counter, "Dev", "multilabel")
write_distribution(path, counter, "Dev")
counter = Counter(tuple(item) for item in test_labels_ref_list)
plot_distribution(counter, "Test", "multilabel")
write_distribution(path, counter, "Test")

# Filter labels keeping only primary ones
total_ref_primary_label = [label[0] for label in total_labels_ref_list]
train_ref_primary_label = [label[0] for label in train_labels_ref_list]
dev_ref_primary_label = [label[0] for label in dev_labels_ref_list]
test_ref_primary_label = [label[0] for label in test_labels_ref_list]

# Only primary label distribution counts + graphs
path= 'output/partition/1labelDistribution.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    print("Distribution of labels\n", file=file)
counter = Counter(total_ref_primary_label)
plot_distribution(counter, "Total", "1label")
write_distribution(path, counter, "Total")
counter = Counter(train_ref_primary_label)
plot_distribution(counter, "Train", "1label")
write_distribution(path, counter, "Train")
counter = Counter(dev_ref_primary_label)
plot_distribution(counter,"Dev", "1label")
write_distribution(path, counter, "Dev")
counter = Counter(test_ref_primary_label)
plot_distribution(counter,"Test", "1label")
write_distribution(path, counter, "Test")
