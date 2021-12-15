from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn import metrics
from utils import clean_corpus, reconstruct_hyphenated_words, write_output_stats_file, write_predictions_file, create_confusion_matrix
from partition import sents_train, labels_train, sents_dev, labels_dev, sents_test, labels_test
import spacy
import numpy
from collections import Counter 
#import matplotlib.pyplot as plt
import os
from nltk.corpus import stopwords

# Transform labels list with names in label array with number representations
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

# Create sparse matrixes that represent words present in each sentence, which is the appropriate format to feed the AI classifier
def format_sentVector_to_SparseMatrix(vectors_list):
    for i, sent_vector in enumerate(vectors_list): 
        sparse_vector = [0] * len(words_to_numbers) # vocabulary size cause each word present is a feature
        counts = Counter(sent_vector)
        for index, freq in counts.items():
            sparse_vector[index] = 1 # freq/len(sent_vector) # DIFFERENT CONFIGURATION POSSIBILITIES
        if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
            matrix_array = [sparse_vector]
        else:
            matrix_array.append(sparse_vector)
    matrix_array = numpy.asarray(matrix_array)
    return matrix_array

# Create sentences representation in numeric format, according to dictionary
def create_vectors_list(sents):
    unk_count = 0
    vectors_list = []
    for sent in sents:
        sent_doc = clean_corpus(sent) 
        sent_doc = nlp(sent_doc)
        sent_doc = reconstruct_hyphenated_words(sent_doc)
        sent_doc = [token.text for token in sent_doc if not token.is_space if not token.is_punct] # if not token.text in stopwords.words()]
        # token i and token i+1
        sent_tokens_list = []
        sent_vector = []
        for token in sent_doc:  
            if token.lower() not in words_to_numbers: 
                sent_tokens_list.append("unk")
                unk_count += 1
                #print(token.lower())
            else:
                sent_tokens_list.append(token.lower())
                #print(token.lower())
            sent_vector = numpy.append(sent_vector, words_to_numbers[sent_tokens_list[-1]])
        if len(sent_vector) > 0:
            sent_vector = sent_vector.astype(int)
        vectors_list.append(sent_vector)
        print(sent_vector)
    print("Unk count:", unk_count)
    return vectors_list

####
# MAIN

nlp = spacy.load('en_core_web_lg',disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']) 

# Preprocessing input 

#print(sents_train)
corpus = ' '.join(sents_train)
corpus = clean_corpus(corpus) 
train_doc = nlp(corpus)
train_doc = reconstruct_hyphenated_words(train_doc)
tokens = [token.text for token in train_doc if not token.is_space if not token.is_punct] # if not token.text in stopwords.words()] 
# FLAG: As extra parameters think of removing  LITTLE SQUARE, SMTH ELSE AS WELL? 

word_freq = Counter(tokens)
#print(word_freq)

# FLAG - checked

# Remove words less frequent than  2 (or equal?)
corpus_with_unk = [word[0] for word in word_freq.items() if int(word[1]) > 2] # < 2 or <= 2
#print(corpus_with_unk)

#### FLAG - REVIEW IF WORD FREQUENCY SHOULD BE COUNTED WITHOUT SPACY TOKENIZATION 
#  FLAG exclusion of all less or equal to 2 correctly - checked
# COUNTING REJOINED TRAIN CORPUS x ORIGINAL SENTENCE TRAIN

# Creating dictionary
words_to_numbers = {}
number_representation = 0
for word in corpus_with_unk:
    words_to_numbers[word] = number_representation
    number_representation += 1
words_to_numbers["unk"] = number_representation

#for i in words_to_numbers:
#    print(i, words_to_numbers[i])
print("Length of the dictionary of word representations:",len(words_to_numbers))

# FLAG - CHECK IF DICTIONARY IS BUILT CORRECTLY
#               SHOULD PUNCTUATION BE UNKNOWN? BECAUSE RIGHT NOW IT IS -NOPE, FIXED
# TO DO: count frequency again?
# count frequency before and after removing unknown words - ??? - ASK GABRIEL!!
# checked that it seems ok

#print(sents_train)
train_vectors_list = create_vectors_list(sents_train)
dev_vectors_list = create_vectors_list(sents_dev)
test_vectors_list = create_vectors_list(sents_test)

# COUNT STATISTICS - HOW MANY WORDS WERE CONSIDERED UNK, AND HOW MANY OF EACH WORD

# FLAG - CHECK IF SENTENCE REPRESENTATIONS WERE DONE CORRECTLY

#for sent in train_vectors_list:
#    print(sent)
#print(train_vectors_list)

train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list)
dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list)
test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list)

# FLAG - CHECK IF SPARSE MATRIX REPRESENTATION WAS DONE CORRECTLY

train_labels_primary = create_labels_array(labels_train)
dev_labels_primary = create_labels_array(labels_dev)
test_labels_primary = create_labels_array(labels_test)

# FLAG - ENSURE THAT LABELS LIST ARE CORRECTLY MADE

# CLASSIFIER

# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=100, learning_rate=0.5) # n_est 25, 50, 75, 100,200, 300 lr 0.5, 1

# FLAG - CHECK WHICH CONFIGURATIONS SHOULD BE HERE - checked

# Choosing best hyperparameters
params = [{'n_estimators': [25, 50, 75, 100, 200, 300], 'learning_rate': [0.5,0.75,0.9,1,1.1,1.2]}]
classifier = GridSearchCV(adaclassifier, params)


# Training
#model = adaclassifier.fit(train_matrix_array, train_labels_primary) 
classifier.fit(train_matrix_array, train_labels_primary) 
#print(classifier.best_params_)
# DECISION TREE, WRONG FUNCTION, DELETE IT
#decision = adaclassifier.decision_function(train_matrix_array)
#print(decision)

# Predicting
#predictions = model.predict(dev_matrix_array)
predictions = classifier.predict(dev_matrix_array)
#predictions = classifier.predict(test_matrix_array)

# casually printing results
#for sent, pred in zip(sents_train,predictions):
#    print(sent, pred, "\n")
#print("Predictions:\n", predictions)

# Confusion matrix
test_list = test_labels_primary.tolist()
dev_list = dev_labels_primary.tolist()
pred_list = [pred for pred in predictions]
labels=[1,3,5,4,2]
path='output/AI Classifier/1Label_confusion_matrix_NormTrue.jpg'
display_labels=['Commit to privacy', 'Declare opinion about privacy','Not applicable','Related to privacy','Violate privacy']
create_confusion_matrix(dev_list, pred_list, "true", path, labels, display_labels)
#create_confusion_matrix(test_list, pred_list, "true", path, labels, display_labels)
path='output/AI Classifier/1Label_confusion_matrix_NonNorm.jpg'
create_confusion_matrix(dev_list, pred_list, None, path, labels, display_labels)
#create_confusion_matrix(test_list, pred_list, None, path, labels, display_labels)

# FLAG - CHECK IF CONFUSION MATRIX IS CORRECT FOR EVERY LABEL

path='output/AI Classifier/1labelPredictionsStatsDev.txt'
#path='output/AI Classifier/1labelPredictionsStatsTest.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    print("Performance measures\n", file=file)
write_output_stats_file(path, "Dev", dev_labels_primary, predictions, labels)
#write_output_stats_file(path, "Test", test_labels_primary, predictions, labels)

# TO DO: WRITE PREDICTIONS JSON FILE -> LEARN HOW TO TRANSFORM ADABOOST OUTPUT IN DICT ( LIST OF ({"text":sentence['text'], "label":label}))
#write_predictions_file("Dev", dev_pred_dict)
#write_predictions_file("Test", test_pred_dict)
# FLAG - CHECK IF THESE ARE THE RIGHT MEASURES, CALCULATED CORRECTLY AND ROUNDED CORRECTLY


# MAKE SURE THAT RESULTS MAKE SENSE, OTHERWISE MAYBE THERE'S A LOST MISTAKE

# EXPERIMENTS

# INITIAL SETTINGS

# STOPWORDS INCLUDED
# PUNCTUATION IS INCLUDED IN VECTORS LIST AND SEEN AS UNKNOWN I THINK
# INCLUDES LITTLE SQUARE AND OTHER WEIRD CHARACTERS
# EXCLUDE WORDS WITH FREQUENCY < 2 (TURN THEM INTO UNKNOWN)
# WEIGHTED VALUES TO EACH WORD ACCORDING TO THEIR FREQUENCY IN THE SENTENCE
# N_ESTIMATORS: 50
# LEARNING RATE: 1
# ALL OTHER PARAMETERs: DEFAULT
# MULTILABEL IS BEING TACKLED BY CONSIDERING ONLY PRIMARY LABEL

# CHANGES

# REMOVE STOP WORDS
# REMOVE PUNCTUATION FROM VECTORS LIST
# REMOVE LITTLE SQUARE AND OTHER ODD CHARACTERS
# INCLUDE WORDS WITH FREQUENCY >= 2  
# TRY WITH 1 INSTEAD OF WEIGHTED VALUE + TRY OTHER WAYS TO REPRESENT THE SENTENCE MAYBE
# DIFFERENT CONFIGURATIONS FOR THE CLASSIFIER (N_ESTIMATORS, LEARNING RATE, ETC?)
# MAKE AVERAGE OF RESULTS SINCE THEY DIFFER?
# WHAT TO DO ABOUT DECLARE OPINION TO PRIVACY? EXPERIMENT WITHOUT IT?
# DIFFERENT WAYS TO TACKLE MULTI-LABEL: DUPLICATE RESULTS, ALIGN RIGHT LABEL FIRST

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
# REPLACE EVERY WORD THAT IS LESS FREQUENT THAN 2 WITH UNK
#
# use counter from collections, already creates a dictionary , then remove words, add unk row

# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit


# First accuracy without weight: 0.47
# First accuracy weighted: 0.43


# TEST 1
# discard unknown - no difference

# TEST 2
#"for loop"

# gridsearchCV  ---- lr 0.5, n estimators 1
# TEST 3

# features importance
# order from highest
# 10 % from the greater than zero
# save them

# evaluate
# ----
 # TEST 4
# remake it with bigrams (sets of 2 adjacent tokens)
# decrease frequency to 1

# TEST 5
# create a joint model with 2 - 10% best

# TEST 6
# INSTEAD OF ADABOOST , LINEAR REGRESSION, NAIVE BAYES


