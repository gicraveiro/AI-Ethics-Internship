# importing the dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter

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

sents_train, sents_test, labels_train, labels_test = train_test_split(sents,labels, test_size=0.2, stratify=labels)

# save a json, separate labels and sents, use a dictionary in python
train_dict = []
for row_id,row in enumerate(sents_train):
    row = re.sub("\n", " ", row)
    train_dict.append({"text":row.strip(), "label":labels_train[row_id]})

sents_dev, sents_test, labels_dev, labels_test = train_test_split(sents_test,labels_test, test_size=0.5, stratify=labels_test)

dev_dict = []
for row_id,row in enumerate(sents_dev):
    row = re.sub("\n", " ", row)
    dev_dict.append({"text":row.strip(), "label":labels_dev[row_id]})
test_dict = []
for row_id,row in enumerate(sents_test):
    row = re.sub("\n", " ", row)
    test_dict.append({"text":row.strip(), "label":labels_test[row_id]})

# create output files and write sentences with labels
path = 'output/partition/multilabeldata_train.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/multilabeldata_train.txt', 'w') as train_file:
    train_file.write(json.dumps(train_dict, indent=4, ensure_ascii=False))
path = 'output/partition/multilabeldata_dev.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/multilabeldata_dev.txt', 'w') as dev_file:
    dev_file.write(json.dumps(dev_dict, indent=4, ensure_ascii=False))
path = 'output/partition/multilabeldata_test.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/multilabeldata_test.txt', 'w') as test_file:
    test_file.write(json.dumps(test_dict, indent=4, ensure_ascii=False))

# COUNTING DISTRIBUTION TO ENSURE IT IS BEING PERFORMED CORRECTLY

# creating list of labels
with open('output/partition/multilabeldata_train.txt', 'r') as train_file:
    json_obj_train = json.loads(train_file.read())
    train_count = []
    for i in json_obj_train:
        item = i['label']
        item = str(item)
        train_count.append(item)
    print("\n",Counter(train_count), "\n")
with open('output/partition/multilabeldata_test.txt', 'r') as test_file:
    json_obj_test = json.loads(test_file.read())
    test_count = []
    for i in json_obj_test:
        item = i['label']
        item = str(item)
        test_count.append(item)
    print(Counter(test_count),"\n")
with open('output/partition/multilabeldata_dev.txt', 'r') as dev_file:
    json_obj_dev = json.loads(dev_file.read())
    dev_count = []
    for i in json_obj_dev:
        item = i['label']
        item = str(item)
        dev_count.append(item)
    print(Counter(dev_count),"\n")

sum = Counter(train_count)+Counter(test_count)+Counter(dev_count)
print(sum)

# printing the proportion of occurrence of each label in comparison to the total number of sentences with that label
i=0
print("\nTrain set")
for item, sum_item in zip(sorted(Counter(train_count)), sorted(Counter(sum))):
    distr_value = Counter(train_count)[item]
    total_value = Counter(sum)[sum_item]
    print(item,  sum_item)
    print(distr_value, "out of", total_value, "samples in the whole dataset")
    print(round(float(distr_value)/float(total_value)*100,2),'\n')

print("Dev set")
for item, sum_item in zip(sorted(Counter(dev_count)), sorted(Counter(sum))):
    distr_value = Counter(dev_count)[item]
    total_value = Counter(sum)[sum_item]
    print(item,  sum_item)
    print(distr_value, "out of", total_value, "samples in the whole dataset")
    print(round(float(distr_value)/float(total_value)*100,2),'\n')

print("Test set")
for item, sum_item in zip(sorted(Counter(test_count)), sorted(Counter(sum))):
    distr_value = Counter(test_count)[item]
    total_value = Counter(sum)[sum_item]
    print(item,  sum_item)
    print(distr_value, "out of", total_value, "samples in the whole dataset")
    print(round(float(distr_value)/float(total_value)*100,2),'\n')