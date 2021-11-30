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

matrix_list = []

# SOLUTION IDEA: MAYBE FIRST CREATE THE LEXICON AND AFTERWARDS CREATE THE SPARSE MATRIXES...
# WAIT, PAUSE, it seems they were be as long as the sentence anyways if I don't do anything about it... study better

# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit

for sent in sents_train:
    sent_doc = nlp(sent)
    sent_tokens_list = []
    sent_vector = []
    data_array_ofmatrix = []
    indptr = [0]
    for token in sent_doc:
        sent_tokens_list.append(token.text)
        if token.text not in words_to_numbers:
            words_to_numbers[token.text] = number_representation
            number_representation += 1
        #sent_vector.append(words_to_numbers[token.text])
        numpy.append(sent_vector, words_to_numbers[token.text])
        data_array_ofmatrix.append(1)
        sent_vector = numpy.append(sent_vector, words_to_numbers[token.text])
    indptr.append(len(sent_vector))
    dataset_tokens.append(sent_tokens_list)
    vectors_list.append(sent_vector) # numpy.asarray()
    #print(sent_vector)
    mat = csr_matrix((data_array_ofmatrix, sent_vector, indptr), dtype=int).toarray()
    #print(mat[0])
    matrix_list.append(mat[0])
    print("NEW",type(mat), len(mat), type(mat[0]), len(mat[0]))
    
# PAUSE
#for entry in vectors_list:
#    data_array_ofmatrix = []
#    indptr = [0]
#    for token in entry:
#        data_array_ofmatrix.append(1)
        
#    mat = csr_matrix((data_array_ofmatrix, sent_vector, indptr), dtype=int).toarray()
#    print(mat)
#    matrix_list.append(mat)
#    print(type(mat), len(mat))
    #corpus.extend(sent_tokens_list)

#freq = nltk.FreqDist(corpus)

#print(vectors_list)
#print(dataset_tokens)
#print(number_representation)
#print (freq)

#print(labels_train, type(labels_train))
#vectorizer = Vectorizer()
#vectorizer.bert(sents_train)
#sent_vectors = vectorizer.vectors
# Reshaping needed... create word embeddings
#sents_train = sents_train.reshape(-1,1)
#print(sent_vectors)

# PRE-PROCESS LABELS VECTOR
# MULTI-LABEL PROBLEM...
# APPROACH 1: CHOOSE PRIMARY LABEL
# APPROACH 2: DUPLICATE RESULTS
# APPROACH 3? : ALIGN RIGHT LABEL FIRST

labels_primary = []

for label in labels_train:
    if label[0] == 'Commit to privacy':
        labels_primary.append(1)
    if label[0] == 'Violate privacy':
        labels_primary.append(2)
    if label[0] == 'Declare opinion about privacy':
        labels_primary.append(3)
    if label[0] == 'Related to privacy':
        labels_primary.append(4)
    if label[0] == 'Not applicable':
        labels_primary.append(5)
  

print(type(matrix_list), len(matrix_list))
print(type(labels_primary), len(labels_primary))




#print(matrix_list)
#print(labels_train)
# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training
model = adaclassifier.fit(matrix_list, labels_primary)


# Predicting
#predictions = model.predict(sents_test)

#print(predictions)

# Measuring results
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED TO OUTLIERS - declare opinion about privacy is a very rare category...