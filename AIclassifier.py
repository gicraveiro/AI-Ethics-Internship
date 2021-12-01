from scipy.sparse import data
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from partition import sents_train, labels_train, sents_test, labels_test
import re
from sent2vec.vectorizer import Vectorizer
import spacy
import numpy
import nltk
from scipy.sparse import csr_matrix
from sklearn import datasets

# Preprocessing input
# Removing \n 
for row_id,row in enumerate(sents_train):
    row = re.sub("\n", " ", row)
    sents_train[row_id] = row
for row_id,row in enumerate(sents_test):
    row = re.sub("\n", " ", row)
    sents_test[row_id] = row

# TOKENIZE, PREPROCESS, CONVERT WORD TOKENS INTO NUMBERS FROM 1 TO N, N IS THE VOACABULARY SIZE
nlp = spacy.load('en_core_web_sm')

#
# REPLACE EVERY WORD THAT IS LESS FREQUENT THAN 3 OR 4 WITH UNK
#

words_to_numbers = {}
number_representation = 0
train_vectors_list = []

for sent in sents_train:
    sent_doc = nlp(sent)
    #sent_tokens_list = []
    sent_vector = []
    for token in sent_doc:
        #sent_tokens_list.append(token.text)
        if token.text not in words_to_numbers:
            words_to_numbers[token.text] = number_representation
            number_representation += 1
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
        #    words_to_numbers[token.text] = number_representation
        #    number_representation += 1
            #sent_vector = numpy.append(sent_vector, len(words_to_numbers)) # REFERENT TO UNKNOWN THAT WILL BE IMPLEMTNED SOON
            pass
        else:
            sent_vector = numpy.append(sent_vector, words_to_numbers[token.text])
            sent_vector = sent_vector.astype(int)
    test_vectors_list.append(sent_vector) 

for i, sent_vector in enumerate(train_vectors_list): 
    sparse_vector = [0] * len(words_to_numbers)
    for index in sent_vector:
        sparse_vector[index] = 1
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        train_matrix_array = [sparse_vector]
    else:
        train_matrix_array = numpy.concatenate((train_matrix_array, [sparse_vector]), axis=0)

for i, sent_vector in enumerate(test_vectors_list): 
    sparse_vector = [0] * len(words_to_numbers)
    for index in sent_vector:
        sparse_vector[index] = 1 # USE FREQUENCY IN THE SENTENCE AS VALUE
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        test_matrix_array = [sparse_vector]
    else:
        test_matrix_array = numpy.concatenate((test_matrix_array, [sparse_vector]), axis=0)


# PRE-PROCESS LABELS VECTOR
# MULTI-LABEL PROBLEM...
# APPROACH 1: CHOOSE PRIMARY LABEL
# APPROACH 2: DUPLICATE RESULTS
# APPROACH 3? : ALIGN RIGHT LABEL FIRST

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

#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#print(X)
#print(y)
#print(matrix_array)
#print(labels_primary)
#print(len(X), type(X), len(y), type(y))
#print(len(matrix_array), type(matrix_array), len(labels_primary), type(labels_primary))

# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training
model = adaclassifier.fit(train_matrix_array, train_labels_primary)

# Predicting
predictions = model.predict(test_matrix_array)

print(predictions)

# Measuring results
print("Accuracy:",metrics.accuracy_score(test_labels_primary, predictions))

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED TO OUTLIERS - declare opinion about privacy is a very rare category...


# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit
