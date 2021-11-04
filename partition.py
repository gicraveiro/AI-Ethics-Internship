# importing the dataset
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import os
import json
from collections import Counter
annotation = pd.read_csv("data/Facebook/Privacy/Annotated/fbAnnotation.csv")
sents = annotation['Sentences'].values
labels = annotation['Labels'].values
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

# create output files
path = 'output/partition/fbdata_train.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_train.txt', 'w') as train_file:
    train_file.write(json.dumps(train_dict, indent=4, ensure_ascii=False))
path = 'output/partition/fbdata_dev.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_dev.txt', 'w') as dev_file:
    dev_file.write(json.dumps(dev_dict, indent=4, ensure_ascii=False))
path = 'output/partition/fbdata_test.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open('output/partition/fbdata_test.txt', 'w') as test_file:
    test_file.write(json.dumps(test_dict, indent=4, ensure_ascii=False))

with open('output/partition/fbdata_train.txt', 'r') as train_file:
    json_obj_train = json.loads(train_file.read())
    train_count = []
    for i in json_obj_train:
        item = i['label']
        train_count.append(item)
    print(Counter(train_count))
with open('output/partition/fbdata_test.txt', 'r') as test_file:
    json_obj_test = json.loads(test_file.read())
    test_count = []
    for i in json_obj_test:
        item = i['label']
        test_count.append(item)
    print(Counter(test_count))
with open('output/partition/fbdata_dev.txt', 'r') as dev_file:
    json_obj_dev = json.loads(dev_file.read())
    dev_count = []
    for i in json_obj_dev:
        item = i['label']
        dev_count.append(item)
    print(Counter(dev_count))
    
sum = Counter(train_count)+Counter(test_count)+Counter(dev_count)
print(sum)

i=0

for item, sum_item in zip(Counter(train_count).items(), Counter(sum).items()):
    print(sum_item)
    print(float(item[1])/float(sum_item[1])*100,'\n')

for item, sum_item in zip(Counter(dev_count).items(), Counter(sum).items()):
    print(sum_item)
    print(float(item[1])/float(sum_item[1])*100,'\n')

for item, sum_item in zip(Counter(test_count).items(), Counter(sum).items()):
    print(sum_item)
    print(float(item[1])/float(sum_item[1])*100,'\n')
    
