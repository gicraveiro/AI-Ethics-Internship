from scipy.sparse import data
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from utils import clean_corpus, reconstruct_hyphenated_words
from partition import sents_train, labels_train, sents_test, labels_test
import re
from sent2vec.vectorizer import Vectorizer
import spacy
import numpy
import nltk
from scipy.sparse import csr_matrix
from sklearn import datasets
from collections import Counter 
# TOKENIZE, PREPROCESS, CONVERT WORD TOKENS INTO NUMBERS FROM 1 TO N, N IS THE VOACABULARY SIZE
nlp = spacy.load('en_core_web_lg') # FIND ALL THE OTHER SPACY.LOAD AND CHANGE TO LG
# Preprocessing input and create lexicon
# Removing \n 

words_to_numbers = {}
number_representation = 0

#print(sents_train)
corpus = '\n'.join(sents_train)
#print(corpus)
corpus = clean_corpus(corpus)
corpus = re.sub("\n", " ", corpus)
corpus = re.sub("  ", " ", corpus)
train_doc = nlp(corpus)
train_doc = reconstruct_hyphenated_words(train_doc)
tokens = [token.text for token in train_doc if not token.is_space if not token.is_punct] # if not token.text in stopwords.words()] 
# FLAG: SHOULD I REMOVE STOPWORDS, LITTLE SQUARE, SMTH ELSE AS WELL? 

word_freq = Counter(tokens)
print(word_freq)



#### FLAG - REVIEW IF WORD FREQUENCY SHOULD BE CALCULATED WITHOUT SPACY TOKENIZATION
#for item in word_freq.items():
#    if item[1] == 1:
#        print(item)
#print(word_freq)
# calculte frequency before and after removing unknown words
# use counter from collections, already creates a dictionary , then remove words, add unk row
''' 
for row_id,row in enumerate(sents_train):
    row = re.sub("\n", " ", row)
    sents_train[row_id] = row
    sent_doc = nlp(sents_train[row_id])
    for token in sent_doc:
        if token.text not in words_to_numbers:
            words_to_numbers[token.text] = number_representation
            number_representation += 1


for row_id,row in enumerate(sents_test):
    row = re.sub("\n", " ", row)
    sents_test[row_id] = row

#
# REPLACE EVERY WORD THAT IS LESS FREQUENT THAN 2 WITH UNK
#

train_vectors_list = []

for sent in sents_train:
    #sent_doc = nlp(sent)
    #sent_tokens_list = []
    #sent_vector = []
    #for token in sent_doc:
        #sent_tokens_list.append(token.text)
        #if token.text not in words_to_numbers:
        #    words_to_numbers[token.text] = number_representation
        #    number_representation += 1
    sent_vector = numpy.append(sent_vector, words_to_numbers[token.text])
    sent_vector = sent_vector.astype(int)
    train_vectors_list.append(sent_vector) 

test_vectors_list = []

for sent in sents_test:
    sent_doc = nlp(sent)
    #sent_tokens_list = []
    sent_vector = []
    for token in sent_doc:
        #sent_tokens_list.append(token.text)
        # HELP HOW TO DEAL WITH UNKNOWN VALUES WHEN CREATING THE DICTIONARY
        if token.text not in words_to_numbers:
            #sent_vector = numpy.append(sent_vector, len(words_to_numbers)) # REFERENT TO UNKNOWN THAT WILL BE IMPLEMTNED SOON
            pass
        else:
            sent_vector = numpy.append(sent_vector, words_to_numbers[token.text])
            sent_vector = sent_vector.astype(int)
    test_vectors_list.append(sent_vector) 

print(len(sents_test))
for i, sent_vector in enumerate(train_vectors_list): 
    sparse_vector = [0] * len(words_to_numbers)
    for index in sent_vector:
        auxList = sent_vector.tolist()
        freq = auxList.count(index)
        sparse_vector[index] = freq/len(sent_vector) # 1
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        train_matrix_array = [sparse_vector]
    else:
        train_matrix_array = numpy.concatenate((train_matrix_array, [sparse_vector]), axis=0)

for i, sent_vector in enumerate(test_vectors_list): 
    sparse_vector = [0] * len(words_to_numbers)
    for index in sent_vector:
        auxList = sent_vector.tolist()
        freq = auxList.count(index)
        sparse_vector[index] = freq/len(sent_vector) # 1
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        test_matrix_array = [sparse_vector]
    else:
        test_matrix_array = numpy.concatenate((test_matrix_array, [sparse_vector]), axis=0)


# PRE-PROCESS LABELS VECTOR
# MULTI-LABEL PROBLEM...
# APPROACH 1: CHOOSE PRIMARY LABEL ----- ADOPTED
# APPROACH 2: DUPLICATE RESULTS
# APPROACH 3? : ALIGN RIGHT LABEL FIRST

# LESS OR EQUAL THAN 2 TIMES

train_labels_primary = []

for label in labels_train:
    if label[0] == 'Commit to privacy':
        train_labels_primary = numpy.append(train_labels_primary,1)
    if label[0] == 'Violate privacy':
        train_labels_primary = numpy.append(train_labels_primary,2)
    if label[0] == 'Declare opinion about privacy':
        train_labels_primary = numpy.append(train_labels_primary,3)
    if label[0] == 'Related to privacy':
        train_labels_primary = numpy.append(train_labels_primary,4)
    if label[0] == 'Not applicable':
        train_labels_primary = numpy.append(train_labels_primary,5)
train_labels_primary = train_labels_primary.astype(int)

test_labels_primary = []

for label in labels_test:
    if label[0] == 'Commit to privacy':
        test_labels_primary = numpy.append(test_labels_primary,1)
    if label[0] == 'Violate privacy':
        test_labels_primary = numpy.append(test_labels_primary,2)
    if label[0] == 'Declare opinion about privacy':
        test_labels_primary = numpy.append(test_labels_primary,3)
    if label[0] == 'Related to privacy':
        test_labels_primary = numpy.append(test_labels_primary,4)
    if label[0] == 'Not applicable':
        test_labels_primary = numpy.append(test_labels_primary,5)
test_labels_primary = test_labels_primary.astype(int)

# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training
model = adaclassifier.fit(train_matrix_array, train_labels_primary)

# Predicting
predictions = model.predict(test_matrix_array)

print(predictions)

# Measuring results
print("Accuracy:",metrics.accuracy_score(test_labels_primary, predictions))
print("Recall micro:",metrics.recall_score(test_labels_primary, predictions, average="micro"))
print("Recall macro:",metrics.recall_score(test_labels_primary, predictions, average="macro"))
print("F1 Score micro:",metrics.f1_score(test_labels_primary, predictions, average="micro"))
print("F1 Score macro:",metrics.f1_score(test_labels_primary, predictions, average="macro"))
print("F1 Score weighted:",metrics.f1_score(test_labels_primary, predictions, average="weighted"))
# CAREFUL
# ADABOOST IS HIGHLY AFFECTED by OUTLIERS - declare opinion about privacy is a very rare category...


# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit


# First accuracy without weight: 0.47
# First accuracy weighted: 0.43


# USE THE DEV SET TO MAKE EXPERIMENTS ON PERFORMANCE OF THE ALGORITHM
# TEST DIFFERENT WAYS TO REPRESENT THE SENTENCE - AND THEN MAYBE START DOING OTHER THINGS
'''