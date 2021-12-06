# importing the dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter

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

# MAIN

# Reads annotation table from file .csv saved locally and creates labels and senences list
annotation = pd.read_csv("data/Facebook/Privacy/Annotated/AnnotatedMultiLabelDataset.csv")
sents = annotation['Sentences'].values
labels1 = annotation['Primary Label'].values
labels2 = annotation['Secondary Label'].values
labels = []
for l1,l2 in zip(labels1,labels2):
    row_labels = []
    row_labels.append(l1)
    if (type(l2) == str):
        row_labels.append(l2)
    labels.append(row_labels)

# FLAG - 
#  CHECK IF CORRECT AND UPDATED FILE IS BEING USED
#  CHECK IF LABELS LIST ARE BEING BUILT CORRECTLY

# Partitions data into 80% trainset and remaining 20%
sents_train, sents_test, labels_train, labels_test = train_test_split(sents,labels, test_size=0.2, stratify=labels, random_state=1111111)

# Partitions remaining 20% into dev set (10%) and test set (10%)
sents_test, sents_dev, labels_test, labels_dev = train_test_split(sents_test,labels_test, test_size=0.5, stratify=labels_test, random_state=1111111)

# save a json, separate labels and sents, use a dictionary in python
train_dict = create_sent_label_dict(sents_train, labels_train)
dev_dict = create_sent_label_dict(sents_dev, labels_dev)
test_dict = create_sent_label_dict(sents_test, labels_test)
#print(train_dict, dev_dict, test_dict)

# FLAG - CHECK IF EACH SENTENCE WAS ASSOCIATED WITH THE RIGHT LABEL

# create output files and write sentences with labels
write_partition_file(train_dict, 'train')
write_partition_file(dev_dict, 'dev')
write_partition_file(test_dict, 'test')

# FLAG - Check if files were written correctly

# COUNTING DISTRIBUTION TO ENSURE IT IS BEING PERFORMED CORRECTLY
train_count = create_label_list('train')
dev_count = create_label_list('dev')
test_count = create_label_list('test')

sum = Counter(train_count)+Counter(test_count)+Counter(dev_count)
print(sum)

# FLAG - Check if counter worked properly

# printing the proportion of occurrence of each label in comparison to the total number of sentences with that label
print_partition_distribution("Train", train_count, sum)
print_partition_distribution("Dev", dev_count, sum)
print_partition_distribution("Test", test_count, sum)

# TO DO: PLOT DISTRIBUTION TRAINSET, TESTSET 

# FLAG - in theory checked, but RECHECK

######
# REDO PARTITION  -- ?? dont remember why...
