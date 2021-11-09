import json
import os
import re # regular expressions

# Read input sentences
path = 'output/partition/fbdata_train.json'

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences = json.loads(document)

# DEBUG PRINTS
#for sentence in json_sentences:
#    print(sentence)
#    print(sentence['text'])
#    print(sentence['label'],'\n\n')
output_dict = []
label = ''
# SIMPLE CLASSIFIER
for sentence in json_sentences:
    label = ''
    flag = re.search(".*privacy.*", sentence['text'])
    if flag:
        label = 'YES'
    output_dict.append({"text":sentence['text'], "label":label})

print(output_dict)

# WRITE OUTPUT
#path = 'output/partition/fbdata_train.txt'
#os.makedirs(os.path.dirname(path), exist_ok=True)
#with open('output/partition/fbdata_train.txt', 'w') as train_file:
#    train_file.write(json.dumps(output_dict, indent=4, ensure_ascii=False))

