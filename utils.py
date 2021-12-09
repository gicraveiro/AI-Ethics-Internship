import re
import os
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import json

# reconstructs hyphen, slash and apostrophes
def reconstruct_hyphenated_words(corpus):
    i = 0
    while i < len(corpus):
        if((corpus[i].text == "-" or corpus[i].text == "/") and corpus[i].whitespace_ == ""): # identify hyphen ("-" inside a word)
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+2]) # merge the first part of the word, the hyphen and the second part of the word            
        elif(corpus[i].text == "’s" and corpus[i-1].whitespace_ == ""):
            with corpus.retokenize() as retokenizer:
                retokenizer.merge(corpus[i-1:i+1])           
        else: 
            i += 1
    return corpus

# noun chunks that correspond to keywords
def reconstruct_noun_chunks(corpus,keywords):
    i = 0
    while i < len(corpus):
        counter = i
        token = corpus[i].text
        for keyword in keywords:
            kw_lower = keyword.lower()
            index = kw_lower.find(token)
            aux = index
            while (aux != -1 and counter < len(corpus)-1 and token != kw_lower):
                counter += 1
                token += ' '+corpus[counter].text
                aux = kw_lower.find(token)
                if(aux == -1):
                    counter -=1
                    token = corpus[i].text
            if(i != counter):
                if(token == kw_lower): 
                    with corpus.retokenize() as retokenizer:
                        retokenizer.merge(corpus[i:counter+1])
                    break 
                else: 
                    counter = i               
        if(i == counter):
            i += 1
    return corpus

def clean_corpus(corpus):
    corpus = corpus.lower()

    corpus = re.sub(" +", " ", corpus)
    corpus = re.sub("(\s+\-)", r" - ", corpus)
    corpus = re.sub("([a-zA-Z]+)([0-9]+)", r"\1 \2", corpus)
    corpus = re.sub("([0-9]+)([a-zA-Z]+)", r"\1 \2", corpus)
    corpus = re.sub("([()!,;\.\?\[\]\|])", r" \1 ", corpus)
    
    return corpus

# For both classifiers

# WRITE OUTPUT STATISTICS FILE
def write_output_stats_file(path, name, ref_labels, pred_labels):
    #path = 'output/Simple Classifier/1labelPredictionsStats_'+name+'.txt'
    #os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as file:
        print(name,"set:\n", file=file)
        print("Accuracy:",round( accuracy_score( ref_labels, pred_labels), 2), file=file)
        print("Precision micro:",round( precision_score( ref_labels, pred_labels, average="micro"), 2), file=file)
        print("Precision macro:",round( precision_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("Recall micro:",round( recall_score( ref_labels, pred_labels, average="micro"),2), file=file)
        print("Recall macro:",round( recall_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("F1 Score micro:",round( f1_score( ref_labels, pred_labels, average="micro"),2), file=file)
        print("F1 Score macro:",round( f1_score( ref_labels, pred_labels, average="macro"),2), file=file)
        print("F1 Score weighted:",round( f1_score(ref_labels, pred_labels, average="weighted"),2), file=file)
        print("\n", file=file)

# WRITE OUTPUT PREDICTIONS IN JSON FORMAT
def write_predictions_file(name, pred_dict):
    path = 'output/Simple Classifier/multilabelPredictions_'+name+'.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open('output/Simple Classifier/multilabelPredictions_'+name+'.json', 'w') as file:
        file.write(json.dumps(pred_dict, indent=4, ensure_ascii=False))
