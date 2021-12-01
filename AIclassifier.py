from scipy.sparse import data
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from partition import sents_train, labels_train, sents_test
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


#y=y.astype('int') #  for warning/error: specify because it is of type object... 
# ValueError: Unknown label type: 'unknown'


# TOKENIZE, PREPROCESS, CONVERT WORD TOKENS INTO NUMBERS FROM 1 TO N, N IS THE VOACABULARY SIZE
nlp = spacy.load('en_core_web_sm')
dataset_tokens = []
#corpus = []
words_to_numbers = {}
number_representation = 0
vectors_list = []

matrix_list = [[]]
indexes = []
total_tokens = []

# SOLUTION IDEA: MAYBE FIRST CREATE THE LEXICON AND AFTERWARDS CREATE THE SPARSE MATRIXES...
# WAIT, PAUSE, it seems they were be as long as the sentence anyways if I don't do anything about it... study better

# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit

for sent in sents_train:
    sent_doc = nlp(sent)
    sent_tokens_list = []
    sent_vector = []
    indptr = [0]
    for token in sent_doc:
        sent_tokens_list.append(token.text)
        if token.text not in words_to_numbers:
            words_to_numbers[token.text] = number_representation
            number_representation += 1
        sent_vector = numpy.append(sent_vector, words_to_numbers[token.text])
        sent_vector = sent_vector.astype(int)
    vectors_list.append(sent_vector) # numpy.asarray()

for i, sent_vector in enumerate(vectors_list): 
    sparse_vector = [0] * len(words_to_numbers)
    for index in sent_vector:
        sparse_vector[index] = 1
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        matrix_array = [sparse_vector]
    else:
        matrix_array = numpy.concatenate((matrix_array, [sparse_vector]), axis=0)


# PRE-PROCESS LABELS VECTOR
# MULTI-LABEL PROBLEM...
# APPROACH 1: CHOOSE PRIMARY LABEL
# APPROACH 2: DUPLICATE RESULTS
# APPROACH 3? : ALIGN RIGHT LABEL FIRST

labels_primary = []

for label in labels_train:
    if label[0] == 'Commit to privacy':
        labels_primary = numpy.append(labels_primary,1)
    if label[0] == 'Violate privacy':
        labels_primary = numpy.append(labels_primary,2)
    if label[0] == 'Declare opinion about privacy':
        labels_primary = numpy.append(labels_primary,3)
    if label[0] == 'Related to privacy':
        labels_primary = numpy.append(labels_primary,4)
    if label[0] == 'Not applicable':
        labels_primary = numpy.append(labels_primary,5)
labels_primary = labels_primary.astype(int)

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
model = adaclassifier.fit(matrix_array, labels_primary)
#model = adaclassifier.fit(X, y)

# Predicting
#predictions = model.predict(sents_test)

#print(predictions)

# Measuring results
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED TO OUTLIERS - declare opinion about privacy is a very rare category...