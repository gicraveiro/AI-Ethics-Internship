# CODE FOR ARTIFICIAL INTELLIGENCE CLASSIFIERS

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import fasttext.util
from utils import clean_corpus, reconstruct_hyphenated_words, write_output_stats_file, create_confusion_matrix, create_sent_label_dict, write_predictions_file
from partition import sents_train, labels_train, sents_dev, labels_dev, sents_test, labels_test
import spacy
import numpy
from collections import Counter 
from sklearn.neural_network import MLPClassifier
import os

# Creating dictionary
def create_dict(lexicon):
    tokens_to_numbers = {}
    number_representation = 0
    for token in lexicon:
        tokens_to_numbers[token] = number_representation
        number_representation += 1
    tokens_to_numbers["unk"] = number_representation
    return tokens_to_numbers

# Transform labels list with names in label array with number representations
def create_labels_array(labels_list):
    labels_array = []
    for label in labels_list:
        if label[0] == 'Commit to privacy':
            labels_array.append(1)
        if label[0] == 'Violate privacy':
            labels_array.append(2)
        if label[0] == 'Declare opinion about privacy':
            labels_array.append(3)
        if label[0] == 'Related to privacy':
            labels_array.append(4)
        if label[0] == 'Not applicable':
            labels_array.append(5)
    return labels_array

# Transform label array with number representations in labels list with names
def converts_to_text(numerical_pred_list):
    text_pred_list = []
    for pred in numerical_pred_list:
        if pred == 1:
            text_pred_list.append('Commit to privacy')
        if pred == 2:
            text_pred_list.append('Violate privacy')
        if pred == 3:
            text_pred_list.append('Declare opinion about privacy')
        if pred == 4:
            text_pred_list.append('Related to privacy')
        if pred == 5:
            text_pred_list.append('Not applicable')

    return text_pred_list

# Create sparse matrixes that represent words present in each sentence, which is the appropriate format to feed the AI classifier
def format_sentVector_to_SparseMatrix(vectors_list, dictionary):
    for i, sent_vector in enumerate(vectors_list): 
        sparse_vector = [0] * len(dictionary) # vocabulary size cause each word present is a feature
        counts = Counter(sent_vector)
        for index, freq in counts.items():
            if len(counts.items()) > 0:
                sparse_vector[index] = 1 #freq/len(sent_vector) #  DIFFERENT CONFIGURATION POSSIBILITIES
        if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
            matrix_array = [sparse_vector]
        else:
            matrix_array.append(sparse_vector)
    matrix_array = numpy.asarray(matrix_array)
    return matrix_array

# Create sentences representation in numeric format, according to dictionary
def create_vectors_list(sents, conversion_dict):
    unk_unigrams_count = 0
    unk_bigrams_count = 0
    unigrams_vector = []
    bigrams_vector = []
    mixed_vector = []
    
    for sent in sents:
        # preprocessing
        sent_doc = clean_corpus(sent) 
        sent_doc = nlp(sent_doc)
        sent_doc = reconstruct_hyphenated_words(sent_doc)
        sent_doc = [token.text for token in sent_doc if not token.is_space if not token.is_punct] # if not token.text in stopwords.words()]
        sent_bigram = []
        for i in range(0, (len(sent_doc)-1)):
            sent_bigram.append(sent_doc[i].lower()+" "+sent_doc[i+1].lower())
        sent_tokens_list = []
        sent_bigrams_list = []
        mixed_tokens_list = []
        sent_unigrams_vector = []
        sent_mixed_vector = []
        sent_bigrams_vector = []
        for token in sent_doc:  
            if token.lower() not in conversion_dict: 
                #sent_tokens_list.append("unk") # TO CONSIDER UNK TOKENS, UNCOMMENT THESE LINES
                #mixed_tokens_list.append("unk")
                unk_unigrams_count += 1
                pass
            else:
                sent_tokens_list.append(token.lower())
                mixed_tokens_list.append(token.lower())
                sent_unigrams_vector = numpy.append(sent_unigrams_vector, conversion_dict[sent_tokens_list[-1]]) # outside else to go back to considering unk 
                sent_mixed_vector = numpy.append(sent_mixed_vector, conversion_dict[token])
            if len(sent_unigrams_vector) > 0:
                sent_unigrams_vector = sent_unigrams_vector.astype(int)
            if len(sent_mixed_vector) > 0:
                sent_mixed_vector = sent_mixed_vector.astype(int)
            
        for bigram in sent_bigram:
            if bigram not in conversion_dict:
                #sent_bigrams_list.append("unk") TO CONSIDER UNK TOKENS, UNCOMMENT THESE LINES
                unk_bigrams_count += 1
                pass
            else:
                sent_bigrams_list.append(bigram)
                mixed_tokens_list.append(bigram)
                sent_bigrams_vector = numpy.append(sent_bigrams_vector, conversion_dict[sent_bigrams_list[-1]])
                sent_mixed_vector = numpy.append(sent_mixed_vector, conversion_dict[bigram])
            if len(sent_bigrams_vector) > 0:
                sent_bigrams_vector = sent_bigrams_vector.astype(int)
            if len(sent_mixed_vector) > 0:
                sent_mixed_vector = sent_mixed_vector.astype(int)
        unigrams_vector.append(sent_unigrams_vector)
        bigrams_vector.append(sent_bigrams_vector)
        mixed_vector.append(sent_mixed_vector)

    print("Unigrams unknown count including repetitions:", unk_unigrams_count)
    print("Bigrams unknown count including repetitions:", unk_bigrams_count, "\n")
    return unigrams_vector  # TO RUN WITH UNIGRAMS, UNCOMMENT THIS LINE AND COMMENT THE OTHER TWO RETURNS
    #return bigrams_vector  # TO RUN WITH BIGRAMS, UNCOMMENT THIS LINE AND COMMENT THE OTHER TWO RETURNS
    #return mixed_vector    # TO RUN WITH UNIGRAMS + BIGRAMS, UNCOMMENT THIS LINE AND COMMENT THE OTHER TWO RETURNS

# TO USE MLP CLASSIFIER WITH WORD EMBEDDINGS APPROACH, UNCOMMENT THIS FUNCION
def create_word_embedding(partition):
    word_embedding_features = []
    for sent in partition:
        sent_doc = clean_corpus(sent) 
        sent_doc = nlp(sent_doc)
        sent_doc = reconstruct_hyphenated_words(sent_doc)
        sent_doc = [token.text for token in sent_doc if not token.is_space if not token.is_punct]
        sentence_embedding = []
        for token in sent_doc:
            token_word_embedding = ft.get_word_vector(token)
            sentence_embedding.append(token_word_embedding)
        we_mean = numpy.asarray(sentence_embedding).mean(axis=0)
        #if isinstance(we_mean, float):
        #    we_mean = numpy.zeros(300, dtype=float)
        word_embedding_features.append(we_mean)
        #word_embedding_features = numpy.asarray(word_embedding_features)
        #word_embedding_features = numpy.append(word_embedding_features, we_mean)
        #tokens_list_of_lists.append(sent_doc)
    #word_embedding_features = numpy.asarray(word_embedding_features)
    word_embedding_features = word_embedding_features
    return word_embedding_features

####
# MAIN

nlp = spacy.load('en_core_web_lg',disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']) 

# Preprocessing 

corpus = ' '.join(sents_train)
corpus = clean_corpus(corpus) 
train_doc = nlp(corpus)
train_doc = reconstruct_hyphenated_words(train_doc)
corpus_in_unigrams = [token.text for token in train_doc if not token.is_space if not token.is_punct] 

# Creating bigram corpus
corpus_in_bigrams = []
for i in range(0,len(corpus_in_unigrams)-1):
    corpus_in_bigrams.append(corpus_in_unigrams[i]+" "+corpus_in_unigrams[i+1])

# Counting frequency of unigrams and bigrams
unigram_freq = Counter(corpus_in_unigrams)
bigram_freq = Counter(corpus_in_bigrams)
# print("Unigrams frequency before removing unknown words:", unigram_freq)
# print("Bigrams frequency before removing unknown words:", bigram_freq)

# Removing unigrams less frequent than 3 and bigrams less frequent than 2
unigrams_filtered_lexicon = [unigram[0] for unigram in unigram_freq.items() if int(unigram[1]) > 2]
bigrams_filtered_lexicon = [bigram[0] for bigram in bigram_freq.items() if int(bigram[1]) > 1] 
# print("Unigrams frequency after removing unknown words:", [unigram for unigram in unigram_freq.items() if int(unigram[1]) > 2])
# print("Bigrams frequency after removing unknown words:", [bigram for bigram in bigram_freq.items() if int(bigram[1]) > 1] )

# Counting unknown tokens
unknown_unigrams = [unigram[0] for unigram in unigram_freq.items() if int(unigram[1]) <= 2]
unknown_bigrams = [bigram[0] for bigram in bigram_freq.items() if int(bigram[1]) <= 1]
print("\n","Unknown unigrams count without repetitions:", len(unknown_unigrams))
print("Unknown bigrams count without repetitions:", len(unknown_bigrams), "\n")

# Unigram dictionary
unigrams_to_numbers = create_dict(unigrams_filtered_lexicon)

# Bigram dictionary
bigrams_to_numbers = create_dict(bigrams_filtered_lexicon)

# Mixed dictionary
with open('data/Utils/features.txt', 'r') as file:
    features_list = file.read()
features_list = features_list.split('\n')
mixed_to_numbers = create_dict(features_list)

print("Length of the dictionary of unigrams(lexicon):",len(unigrams_to_numbers))
print("Length of the dictionary of bigrams(lexicon):",len(bigrams_to_numbers))
print("Length of the dictionary of unigrams and bigrams(lexicon):",len(mixed_to_numbers), "\n")

# CREATE SENTENCE REPRESENTATIONS
#   can either be by word embeddings or with a simple representation according to the presence of a unigram or bigram in the sentence

# WORD EMBEDDINGS

# TO USE MLP CLASSIFIER WITH WORD EMBEDDINGS APPROACH, UNCOMMENT THIS LINE
ft = fasttext.load_model('cc.en.300.bin')

train_word_embedding_features = create_word_embedding(sents_train)
dev_word_embedding_features = create_word_embedding(sents_dev)
test_word_embedding_features = numpy.asarray(create_word_embedding(sents_test))

# SIMPLE NUMERICAL REPRESENTATIONS OF THE SENTENCES

# TO RUN WITH UNIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
train_vectors_list = create_vectors_list(sents_train, unigrams_to_numbers)
dev_vectors_list = create_vectors_list(sents_dev, unigrams_to_numbers)
test_vectors_list = create_vectors_list(sents_test, unigrams_to_numbers)

# TO RUN WITH BIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
# train_vectors_list = create_vectors_list(sents_train, bigrams_to_numbers)
# dev_vectors_list = create_vectors_list(sents_dev, bigrams_to_numbers)
# test_vectors_list = create_vectors_list(sents_test, bigrams_to_numbers)

# TO RUN WITH UNIGRAMS + BIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
# train_vectors_list = create_vectors_list(sents_train, mixed_to_numbers)
# dev_vectors_list = create_vectors_list(sents_dev, mixed_to_numbers)
# test_vectors_list = create_vectors_list(sents_test, mixed_to_numbers)

# FORMATTING SIMPLE SENTENCE REPRESENTATIONS - MUST BE IN SPARSE MATRIX FORMAT TO FEED THE CLASSIFIERS

# TO RUN WITH UNIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, unigrams_to_numbers)
dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, unigrams_to_numbers)
test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, unigrams_to_numbers)

# TO RUN WITH BIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
# train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, bigrams_to_numbers)
# dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, bigrams_to_numbers)
# test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, bigrams_to_numbers)

# TO RUN WITH UNIGRAMS + BIGRAMS, UNCOMMENT THIS 3 LINES AND COMMENT THE OTHER TWO TRIPLETS
# train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, mixed_to_numbers)
# dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, mixed_to_numbers)
# test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, mixed_to_numbers)

# CREATE LABELS REPRESENTATIONS

train_labels_primary = create_labels_array(labels_train)
#dev_labels_primary = numpy.asarray(create_labels_array(labels_dev)) # USE THIS LINE FOR ALL OTHER CLASSIFIERS
dev_labels_primary = create_labels_array(labels_dev) # USE THIS LINE FOR MLP
test_labels_primary = numpy.asarray(create_labels_array(labels_test))

# CLASSIFIER Configurations

# ADABOOST
# Used to choose best hyperparameters for Adaboost classifier
# TO TEST PARAMETERS FOR ADABOOST CLASSIFIER, UNCOMMENT THIS SECTION AND COMMENT OTHER MODELS
#adaclassifier = AdaBoostClassifier() # n_est 25, 50, 75, 100,200, 300 lr 0.5, 1
#params = [{'n_estimators': [25, 50, 75, 100, 200, 300], 'learning_rate': [0.5,0.75,0.9,1,1.1,1.2]}]
#adaclassifier = GridSearchCV(adaclassifier, params)

# Used to choose best hyperparameters for MLP classifier
# TO TEST HYPERPARAMETERS FOR MLP CLASSIFIER, UNCOMMENT THIS SECTION
# parameter_space = {
#     'activation': ['tanh', 'relu'], # 'identity', 'logistic',
#     'solver': ['adam'], #, 'lbfgs'],
#     'learning_rate_init': [0.001], # [0.0001, 0.001, 0.01, 0.05], # 0.1
#     'learning_rate': ['adaptive', 'constant'],
#     'hidden_layer_sizes': [(200,200,200),(200, 250, 200), (250, 200, 250), (250, 250, 250)], # (50,) ,(400,400,400) (150, 150, 150), (200,150,200), (150, 200, 150),
#     'max_iter': [5200],
# #    'early_stopping': ['True', 'False']
# }

#params = {'activation': 'relu', 'hidden_layer_sizes': (200, 250, 200), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 5200, 'solver': 'adam'}

# Classifier models

# TO USE ADABOOST CLASSIFIER, UNCOMMENT adaclassifier AND COMMENT OTHER MODELS
#   TO USE UNIGRAMS, PARAMETERS WERE BETTER WITH n_estimators=50, learning_rate=1
#   TO USE UNIGRAMS + BIGRAMS, PARAMETERS WERE BETTER WITH n_estimators=100, learning_rate=0.5
# adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1) 
# TO USE SVC CLASSIFIER WITH ONE VS REST SCHEME, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
# svc_classifier = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC(dual=False,random_state=None, tol=1e-5, C=0.05)))
# TO USE SVC CLASSIFIER WITH ONE VS ONE SCHEME, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
# svc_classifier = make_pipeline(StandardScaler(), OneVsOneClassifier(LinearSVC(dual=False,random_state=None, tol=1e-5, C=0.05)))
# TO USE MLP CLASSIFIER WITH WORD EMBEDDINGS, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
mlp_classifier = MLPClassifier(random_state=1111111, early_stopping=True, batch_size=32, hidden_layer_sizes=(200,250,200), learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000)
# TO TEST HYPERPARAMETERS FOR MLP CLASSIFIER, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
#opt_mlp = GridSearchCV(mlp_classifier, parameter_space, n_jobs=-1, cv=10) 

# Training

# TO USE ADABOOST CLASSIFIER, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
# model = adaclassifier.fit(train_matrix_array, train_labels_primary) 
#print(adaclassifier.best_params_) # prints best parameters if you enabled GridSearchCV
# TO USE SVC CLASSIFIER, UNCOMMENT THIS LINE AND COMMENT OTHER MODELS
# model = svc_classifier.fit(train_matrix_array, train_labels_primary)

# TO USE MLP CLASSIFIER, UNCOMMENT THESE 3 LINES AND COMMENT OTHER MODELS
new_train_features = numpy.asarray(train_word_embedding_features + dev_word_embedding_features)
new_train_labels = numpy.asarray(train_labels_primary + dev_labels_primary)
model = mlp_classifier.fit(new_train_features, new_train_labels)

# TO TEST PARAMETERS FOR MLP CLASSIFIER UNCOMMENT THESE 2 LINES AND COMMENT OTHER MODELS
#model = opt_mlp.fit(new_train_features, new_train_labels)
#print(model.best_params_)

# TO SEE WHICH FEATURES ADABOOST CHOSE, UNCOMMENT THIS SECTION
# importances = model.feature_importances_
# features = {}
# UNCOMMENT THE LINE YOU NEED FROM THESE 3 AND COMMENT THE OTHER 2
# for i,(token,value) in enumerate(zip(unigrams_to_numbers, importances)):
# # for i,(token,value) in enumerate(zip(mixed_to_numbers, importances)):
# #for i,(token,value) in enumerate(zip(bigrams_to_numbers, importances)): 
#    if (value != 0):
#        features[token] = value
# features = sorted([(value, key) for (key, value) in features.items()], reverse=True)
# for feature in features:
#    print('Feature:',feature[1],'Score:',feature[0])

# Predicting

# TO USE ADABOOST OR SVC CLASSIFIERS
#predictions = model.predict(dev_matrix_array) # DEV
#predictions = model.predict(test_matrix_array) # TEST

# TO USE MLP CLASSIFIER WITH WORD EMBEDDINGS
predictions = model.predict(test_word_embedding_features) # TEST

# Output of results

# Format labels and predictions
test_list = test_labels_primary.tolist() # TEST
#dev_list = dev_labels_primary.tolist() # DEV - ALWAYS COMMENT THIS LINE FOR MLP
pred_list = [pred for pred in predictions]
labels=[1,3,5,4,2]

# path = 'output/AI Classifier/Adaboost/Unigrams/' 
# path = 'output/AI Classifier/Adaboost/Unigrams+Bigrams/'
# path = 'output/AI Classifier/SVC/One vs Rest/C=1/'
# path = 'output/AI Classifier/SVC/One vs One/C=1/'
# path = 'output/AI Classifier/SVC/One vs Rest/C=0.05/'
# path = 'output/AI Classifier/SVC/One vs One/C=0.05/'
# path = 'output/AI Classifier/SVC/C=0.05/'
path = 'output/AI Classifier/MLP/'

# aux_path= path + 'confusion_matrix_Normalized_Dev.png' 
aux_path= path + 'confusion_matrix_Normalized_Test.png' 
display_labels=['Commit to privacy', 'Declare opinion about privacy','Not applicable','Related to privacy','Violate privacy']

# NORMALIZED CONFUSION MATRIX
#create_confusion_matrix(dev_list, pred_list, "true", aux_path, labels, display_labels) # DEV
create_confusion_matrix(test_list, pred_list, "true", aux_path, labels, display_labels) # TEST

# NON NORMALIZED CONFUSION MATRIX
#aux_path= path + 'confusion_matrix_NonNormalized_Dev.png'
aux_path= path + 'confusion_matrix_NonNormalized_Test.png'
#create_confusion_matrix(dev_list, pred_list, None, aux_path, labels, display_labels) # DEV
create_confusion_matrix(test_list, pred_list, None, aux_path, labels, display_labels) # TEST

# File for performance on predictions
#aux_path = path + 'PredictionsStatsDev.txt' # DEV
aux_path = path + 'PredictionsStatsTest.txt' # TEST

with open(aux_path, 'w') as file:
    print("Performance measures - MLP Word Embeddings\n", file=file) # CHANGE TITLE ACCORDING TO CONTEXT
    #print("Performance measures - SVC\n0ne vs Rest, C = 0.05\n", file=file) # CHANGE TITLE ACCORDING TO CONTEXT
#write_output_stats_file(aux_path, "Dev", dev_labels_primary, predictions, labels) # DEV
write_output_stats_file(aux_path, "Test", test_labels_primary, predictions, labels) # TEST


pred_list = converts_to_text(pred_list)
#dev_pred_dict = create_sent_label_dict(sents_dev, pred_list) # DEV
test_pred_dict = create_sent_label_dict(sents_test, pred_list) # TEST

# Predictions json file
#aux_path = path + 'Predictions_Dev.json'
aux_path = path + 'Predictions_Test.json'

#write_predictions_file(dev_pred_dict, aux_path) # DEV
write_predictions_file(test_pred_dict, aux_path) # TEST

# References that helped me understand how to implement all this
# https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit