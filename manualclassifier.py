import json

path = 'output/partition/fbdata_train.json'

json_sentences = []

with open(path) as file:
    document = file.read()
    json_sentences = json.loads(document)

for sentence in json_sentences:
    #print(sentence)
    print(sentence['text'])