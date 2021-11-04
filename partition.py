# importing the dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os

annotation = pd.read_csv("data/Facebook/Privacy/Annotated/fbAnnotation.csv")
sents = annotation['Sentences'].values
labels = annotation['Labels'].values
sents_train, sents_test, labels_train, labels_test = train_test_split(sents,labels, test_size=0.2, stratify=labels)

# save a json, separate labels and sents, use a dictionary in python
train_dict = []
for row_id,row in enumerate(sents_train):
    row = re.sub("\n", " ", row)
    train_dict.append({"text":row, "label":labels_train[row_id]})

sents_dev, sents_test, labels_dev, labels_test = train_test_split(sents_test,labels_test, test_size=0.5, stratify=labels_test)

dev_dict = []
for row_id,row in enumerate(sents_dev):
    row = re.sub("\n", " ", row)
    dev_dict.append({"text":row, "label":labels_dev[row_id]})
test_dict = []
for row_id,row in enumerate(sents_test):
    row = re.sub("\n", " ", row)
    test_dict.append({"text":row, "label":labels_test[row_id]})

# create output files
path = 'output/partition/fbdata_train.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_train.txt', 'w') as train_file:
    print(train_dict, file=train_file)
path = 'output/partition/fbdata_dev.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_dev.txt', 'w') as dev_file:
    print(dev_dict, file=dev_file)
path = 'output/partition/fbdata_test.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_test.txt', 'w') as test_file:
    print(test_dict, file=test_file)