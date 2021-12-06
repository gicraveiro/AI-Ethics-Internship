from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from utils import clean_corpus, reconstruct_hyphenated_words
from partition import sents_train, labels_train, sents_test, labels_test
import re
import spacy
import numpy
from collections import Counter 
import matplotlib.pyplot as plt

# TRANSFORM LABELS LIST WITH NAMES IN LABEL ARRAY WITH NUMBER REPRESENTATIONS
def create_labels_array(labels_list):
    labels_array = []
    for label in labels_list:
        if label[0] == 'Commit to privacy':
            labels_array = numpy.append(labels_array,1)
        if label[0] == 'Violate privacy':
            labels_array = numpy.append(labels_array,2)
        if label[0] == 'Declare opinion about privacy':
            labels_array = numpy.append(labels_array,3)
        if label[0] == 'Related to privacy':
            labels_array = numpy.append(labels_array,4)
        if label[0] == 'Not applicable':
            labels_array = numpy.append(labels_array,5)
    labels_array = labels_array.astype(int)
    return labels_array

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
#print(word_freq)

#
# REPLACE EVERY WORD THAT IS LESS FREQUENT THAN 2 WITH UNK
#

#for word in word_freq.items():
#    print(word[0], word[1])
corpus_with_unk = [word[0] for word in word_freq.items() if int(word[1]) > 2] # < 2 or <= 2
#print(corpus_with_unk)


#### FLAG - REVIEW IF WORD FREQUENCY SHOULD BE COUNTED WITHOUT SPACY TOKENIZATION

for word in corpus_with_unk:
    words_to_numbers[word] = number_representation
    number_representation += 1
words_to_numbers["unk"] = number_representation

# REVIEW DICTIONARY

# TO DO: count frequency again?

# count frequency before and after removing unknown words - ???

# use counter from collections, already creates a dictionary , then remove words, add unk row


#for row_id,row in enumerate(sents_test):
#    row = re.sub("\n", " ", row)
#    sents_test[row_id] = row
train_vectors_list = []

for sent in sents_train:
    sent_doc = nlp(sent)
    sent_tokens_list = []
    sent_vector = []
    for token in sent_doc:  
        if token.text not in words_to_numbers:
            sent_tokens_list.append("unk")
        else:
            sent_tokens_list.append(token.text)
        sent_vector = numpy.append(sent_vector, words_to_numbers[sent_tokens_list[-1]])
    sent_vector = sent_vector.astype(int)
    train_vectors_list.append(sent_vector) 

#print(train_vectors_list)

test_vectors_list = []

for sent in sents_test:
    sent_doc = nlp(sent)
    sent_tokens_list = []
    sent_vector = []
    for token in sent_doc:  
        if token.text not in words_to_numbers:
            sent_tokens_list.append("unk")
        else:
            sent_tokens_list.append(token.text)
        sent_vector = numpy.append(sent_vector, words_to_numbers[sent_tokens_list[-1]])
    sent_vector = sent_vector.astype(int)
    test_vectors_list.append(sent_vector) 


# FLAG - CHECK IF SENTENCE REPRESENATIONS WERE DONE CORRECTLY

#print(test_vectors_list)
#print(len(sents_test))


for i, sent_vector in enumerate(train_vectors_list): 
    sparse_vector = [0] * len(words_to_numbers) # vocabulary size cause each word present is a feature
    for index in sent_vector:
        auxList = sent_vector.tolist()
        freq = auxList.count(index)
        sparse_vector[index] = freq/len(sent_vector) # 1 # LATER TEST AND DOCUMENT PERFORMANCE IN WEIGHTED AND SIMPLE 1
    if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
        train_matrix_array = [sparse_vector]
    else:
        train_matrix_array = numpy.concatenate((train_matrix_array, [sparse_vector]), axis=0)

# TO DO: OPTIMIZE - INSERT IN FUNCTION
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



train_labels_primary = create_labels_array(labels_train)
test_labels_primary = create_labels_array(labels_test)

# FLAG - ENSURE THAT LABELS LIST ARE CORRECTLY MADE + MOVE IT TO FUNCTION

# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# FLAG - CHECK WHICH CONFIGURATIONS SHOULD BE HERE

# Training
model = adaclassifier.fit(train_matrix_array, train_labels_primary)

# Predicting
predictions = model.predict(test_matrix_array)

# casually printing results
#for sent, pred in zip(sents_train,predictions):
#    print(sent, pred, "\n")
#print("Predictions:\n", predictions)

# Confusion matrix
test_list = test_labels_primary.tolist()
pred_list = [pred for pred in predictions]
metrics.ConfusionMatrixDisplay.from_predictions(test_list,pred_list, normalize="true", labels=[1,2,3,4,5],display_labels=['Commit to privacy', 'Violate privacy', 'Declare opinion about privacy', 'Related to privacy', 'Not applicable'])
plt.xticks(rotation=45, ha="right")
plt.subplots_adjust(bottom=0.4)
#plt.show()
plt.savefig('output/AI Classifier/1Label_confusion_matrix.jpg')

# HELP - Predictions are changing... - confusion matrix, and measures

# Measuring results
print("Accuracy:",round(metrics.accuracy_score(test_labels_primary, predictions), 2))
print("Precision micro:",round(metrics.precision_score(test_labels_primary, predictions, average="micro"), 2))
print("Precision macro:",round(metrics.precision_score(test_labels_primary, predictions, average="macro"),2))
print("Recall micro:",round(metrics.recall_score(test_labels_primary, predictions, average="micro"),2))
print("Recall macro:",round(metrics.recall_score(test_labels_primary, predictions, average="macro"),2))
print("F1 Score micro:",round(metrics.f1_score(test_labels_primary, predictions, average="micro"),2))
print("F1 Score macro:",round(metrics.f1_score(test_labels_primary, predictions, average="macro"),2))
print("F1 Score weighted:",round(metrics.f1_score(test_labels_primary, predictions, average="weighted"),2))

# FLAG - CHECK IF THESE ARE THE RIGHT MEASURES, CALCULATED CORRECTLY AND ROUNDED CORRECTLY
# UPLOAD RESULTS IN A DOCUMENT THAT GABRIEL CAN CHECK

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED by OUTLIERS - declare opinion about privacy is a very rare category...


# USE THE DEV SET TO MAKE EXPERIMENTS ON PERFORMANCE OF THE ALGORITHM
# TEST DIFFERENT WAYS TO REPRESENT THE SENTENCE - AND THEN MAYBE START DOING OTHER THINGS

# PRE-PROCESS LABELS VECTOR
# MULTI-LABEL PROBLEM...
# APPROACH 1: CHOOSE PRIMARY LABEL ----- ADOPTED
# APPROACH 2: DUPLICATE RESULTS
# APPROACH 3? : ALIGN RIGHT LABEL FIRST

# HELP - Predictions are changing... - confusion matrix, and measures

# LESS OR EQUAL THAN 2 TIMES

# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit


# First accuracy without weight: 0.47
# First accuracy weighted: 0.43