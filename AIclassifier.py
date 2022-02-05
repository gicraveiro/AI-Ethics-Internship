# CODE FOR ARTIFICIAL INTELLIGENCE CLASSIFIERS

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import fasttext.util
from utils import clean_corpus, reconstruct_hyphenated_words, write_output_stats_file, create_confusion_matrix # write_predictions_file
from partition import sents_train, labels_train, sents_dev, labels_dev, sents_test, labels_test
import spacy
import numpy
from collections import Counter 
from sklearn.neural_network import MLPClassifier
import os

# Creating dictionary
def create_dict(lexicon):
    #print(lexicon)
    tokens_to_numbers = {}
    number_representation = 0
    for token in lexicon:
        #print(token, number_representation)
        tokens_to_numbers[token] = number_representation
        number_representation += 1
    tokens_to_numbers["unk"] = number_representation
    #print(tokens_to_numbers)
    return tokens_to_numbers

# Transform labels list with names in label array with number representations
def create_labels_array(labels_list):
    labels_array = []
    for label in labels_list:
        if label[0] == 'Commit to privacy':
            #labels_array = numpy.append(labels_array,1)
            labels_array.append(1)
        if label[0] == 'Violate privacy':
            #labels_array = numpy.append(labels_array,2)
            labels_array.append(2)
        if label[0] == 'Declare opinion about privacy':
            #labels_array = numpy.append(labels_array,3)
            labels_array.append(3)
        if label[0] == 'Related to privacy':
            #labels_array = numpy.append(labels_array,4)
            labels_array.append(4)
        if label[0] == 'Not applicable':
            #labels_array = numpy.append(labels_array,5)
            labels_array.append(5)
    #labels_array = labels_array.astype(int)
    return labels_array

# Create sparse matrixes that represent words present in each sentence, which is the appropriate format to feed the AI classifier
def format_sentVector_to_SparseMatrix(vectors_list, dictionary):
    for i, sent_vector in enumerate(vectors_list): 
        sparse_vector = [0] * len(dictionary) # vocabulary size cause each word present is a feature
        counts = Counter(sent_vector)
        for index, freq in counts.items():
            if len(counts.items()) > 0:
                sparse_vector[index] = 1 #freq/len(sent_vector) # DIFFERENT CONFIGURATION POSSIBILITIES # 1
        if (i == 0): # TO DO: OPTIMIZE, NO NEED TO CHECK THIS EVERY TURN
            matrix_array = [sparse_vector]
        else:
            matrix_array.append(sparse_vector)
    matrix_array = numpy.asarray(matrix_array)
    return matrix_array

# Create sentences representation in numeric format, according to dictionary
def create_vectors_list(sents, conversion_dict):
    unk_count = 0
    vectors_list = []
    bigrams_vector = []
    mixed_vector = []
    
    for sent in sents:
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
        sent_vector = []
        sent_mixed_vector = []
        sent_bigrams_vector = []
        for token in sent_doc:  
            if token.lower() not in conversion_dict: 
                #sent_tokens_list.append("unk")
                #mixed_tokens_list.append("unk")
                #unk_count += 1
                pass
            else:
                sent_tokens_list.append(token.lower())
                mixed_tokens_list.append(token.lower())
                sent_vector = numpy.append(sent_vector, conversion_dict[sent_tokens_list[-1]]) # outside else to go back to considering unk 
                sent_mixed_vector = numpy.append(sent_mixed_vector, conversion_dict[token])
            if len(sent_vector) > 0:
                sent_vector = sent_vector.astype(int)
            if len(sent_mixed_vector) > 0:
                sent_mixed_vector = sent_mixed_vector.astype(int)
            
            
        for bigram in sent_bigram:
            if bigram not in conversion_dict:
                sent_bigrams_list.append("unk")
                #unk_count += 1
                #pass
            else:
                sent_bigrams_list.append(bigram)
                mixed_tokens_list.append(bigram)
                sent_bigrams_vector = numpy.append(sent_bigrams_vector, conversion_dict[sent_bigrams_list[-1]])
                sent_mixed_vector = numpy.append(sent_mixed_vector, conversion_dict[bigram])
            if len(sent_bigrams_vector) > 0:
                sent_bigrams_vector = sent_bigrams_vector.astype(int)
            if len(sent_mixed_vector) > 0:
                sent_mixed_vector = sent_mixed_vector.astype(int)
        vectors_list.append(sent_vector)
        bigrams_vector.append(sent_bigrams_vector)
        mixed_vector.append(sent_mixed_vector)

    #return vectors_list
    return bigrams_vector
    #return mixed_vector

# def create_word_embedding(partition):

#     word_embedding_features = []
#     for sent in partition:
#         sent_doc = clean_corpus(sent) 
#         sent_doc = nlp(sent_doc)
#         sent_doc = reconstruct_hyphenated_words(sent_doc)
#         sent_doc = [token.text for token in sent_doc if not token.is_space if not token.is_punct]
#         sentence_embedding = []
#         for token in sent_doc:
#             token_word_embedding = ft.get_word_vector(token)
#             sentence_embedding.append(token_word_embedding)
#         we_mean = numpy.asarray(sentence_embedding).mean(axis=0)
#         #if isinstance(we_mean, float):
#         #    we_mean = numpy.zeros(300, dtype=float)
#         word_embedding_features.append(we_mean)
#         #word_embedding_features = numpy.asarray(word_embedding_features)
#         #word_embedding_features = numpy.append(word_embedding_features, we_mean)
#         #tokens_list_of_lists.append(sent_doc)
#     #word_embedding_features = numpy.asarray(word_embedding_features)
#     word_embedding_features = word_embedding_features
#     return word_embedding_features

####
# MAIN

nlp = spacy.load('en_core_web_lg',disable=['tok2vec', 'tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer']) 

# Preprocessing input 

corpus = ' '.join(sents_train)
corpus = clean_corpus(corpus) 
train_doc = nlp(corpus)
train_doc = reconstruct_hyphenated_words(train_doc)
tokens = [token.text for token in train_doc if not token.is_space if not token.is_punct] # if not token.text in stopwords.words()] 
# OBS: MAYBE ENHANCING PREPROCESSING BY REMOVING LITTLE SQUARES COULD BE AN OPTION

corpus_in_bigrams = []
for i in range(0,len(tokens)-1):
    corpus_in_bigrams.append(tokens[i]+" "+tokens[i+1])

token_freq = Counter(tokens)
bigram_freq = Counter(corpus_in_bigrams)

# 

# Remove words less frequent than  2 (or equal?)
corpus_without_unk = [token[0] for token in token_freq.items() if int(token[1]) > 2] # < 2 or <= 2
bigrams_filtered_lexicon = [bigram[0] for bigram in bigram_freq.items() if int(bigram[1]) > 1]

#### FLAG - REVIEW IF WORD FREQUENCY SHOULD BE COUNTED WITHOUT SPACY TOKENIZATION 

# Unigram dictionary
words_to_numbers = create_dict(corpus_without_unk)
# Bigram dictionary
bigrams_to_numbers = create_dict(bigrams_filtered_lexicon)

# Mixed dictionary
# with open('features.txt', 'r') as file:
#     features_list = file.read()
# features_list = features_list.split('\n')
# mixed_to_numbers = create_dict(features_list)

# WORD EMBEDDINGS FOR NN APPROACH
#ft = fasttext.load_model('cc.en.300.bin')


#train_word_embedding_features = create_word_embedding(sents_train)
#dev_word_embedding_features = create_word_embedding(sents_dev)
#test_word_embedding_features = numpy.asarray(create_word_embedding(sents_test))
#print("Length of the dictionary of word representations:",len(words_to_numbers))
print("Length of the dictionary of word representations:",len(bigrams_to_numbers))
#print("Length of the dictionary of word representations:",len(mixed_to_numbers))

# FLAG - CHECK IF DICTIONARY IS BUILT CORRECTLY
#               SHOULD PUNCTUATION BE UNKNOWN? BECAUSE RIGHT NOW IT IS -NOPE, FIXED
# TO DO: count frequency again?
# count frequency before and after removing unknown words - ??? - ASK GABRIEL!!
# checked that it seems ok

# train_vectors_list = create_vectors_list(sents_train, words_to_numbers)
# dev_vectors_list = create_vectors_list(sents_dev, words_to_numbers)
# test_vectors_list = create_vectors_list(sents_test, words_to_numbers)

train_vectors_list = create_vectors_list(sents_train, bigrams_to_numbers)
dev_vectors_list = create_vectors_list(sents_dev, bigrams_to_numbers)
test_vectors_list = create_vectors_list(sents_test, bigrams_to_numbers)

#train_vectors_list = create_vectors_list(sents_train, mixed_to_numbers)
#dev_vectors_list = create_vectors_list(sents_dev, mixed_to_numbers)
#test_vectors_list = create_vectors_list(sents_test, mixed_to_numbers)

# COUNT STATISTICS - HOW MANY WORDS WERE CONSIDERED UNK, AND HOW MANY OF EACH WORD

# FLAG - CHECK IF SENTENCE REPRESENTATIONS WERE DONE CORRECTLY

# train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, words_to_numbers)
# dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, words_to_numbers)
# test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, words_to_numbers)

train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, bigrams_to_numbers)
dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, bigrams_to_numbers)
test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, bigrams_to_numbers)

#train_matrix_array = format_sentVector_to_SparseMatrix(train_vectors_list, mixed_to_numbers)
#dev_matrix_array = format_sentVector_to_SparseMatrix(dev_vectors_list, mixed_to_numbers)
#test_matrix_array = format_sentVector_to_SparseMatrix(test_vectors_list, mixed_to_numbers)

# FLAG - CHECK IF SPARSE MATRIX REPRESENTATION WAS DONE CORRECTLY

train_labels_primary = create_labels_array(labels_train)
dev_labels_primary = numpy.asarray(create_labels_array(labels_dev)) #dev_labels_primary = create_labels_array(labels_dev)
test_labels_primary = numpy.asarray(create_labels_array(labels_test))

# FLAG - ENSURE THAT LABELS LIST ARE CORRECTLY MADE

# CLASSIFIER Configurations

# ADABOOST
# Choosing best hyperparameters
#adaclassifier = AdaBoostClassifier() # n_est 25, 50, 75, 100,200, 300 lr 0.5, 1
#params = [{'n_estimators': [25, 50, 75, 100, 200, 300], 'learning_rate': [0.5,0.75,0.9,1,1.1,1.2]}]
#classifier = GridSearchCV(adaclassifier, params)

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


adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)
#svc_classifier = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC(dual=False,random_state=None, tol=1e-5, C=1)))
#svc_classifier = make_pipeline(StandardScaler(), OneVsOneClassifier(LinearSVC(dual=False,random_state=None, tol=1e-5, C=1)))
#mlp_classifier = MLPClassifier( max_iter=300, early_stopping=True, hidden_layer_sizes=300, batch_size=32) # random_state=1111111,
#mlp_classifier = MLPClassifier(random_state=1111111, early_stopping=True, batch_size=32, hidden_layer_sizes=(200,250,200), learning_rate='adaptive', learning_rate_init=0.001, max_iter=1000)
#opt_mlp = GridSearchCV(mlp_classifier, parameter_space, n_jobs=-1, cv=10)

# Training
model = adaclassifier.fit(train_matrix_array, train_labels_primary) 
#model = classifier.fit(train_matrix_array, train_labels_primary) 
#print(classifier.best_params_)
#model = svc_classifier.fit(train_matrix_array, train_labels_primary)
#new_train_features = numpy.asarray(train_word_embedding_features + dev_word_embedding_features)
#new_train_labels = numpy.asarray(train_labels_primary + dev_labels_primary)
#model = mlp_classifier.fit(new_train_features, new_train_labels)
#model = opt_mlp.fit(new_train_features, new_train_labels)
#print(model.best_params_)

importances = model.feature_importances_

features = {}
# for i,(token,value) in enumerate(zip(words_to_numbers, importances)):
#for i,(token,value) in enumerate(zip(mixed_to_numbers, importances)):
for i,(token,value) in enumerate(zip(bigrams_to_numbers, importances)): # IMPORTANTO TO CHANGE TO ADEQUATE DICT
   if (value != 0):
       features[token] = value
features = sorted([(value, key) for (key, value) in features.items()], reverse=True)
print(features)
for feature in features:
   print('Feature:',feature[1],'Score:',feature[0])

# Predicting
#predictions = model.predict(dev_matrix_array)
predictions = model.predict(test_matrix_array)
#predictions = model.predict(dev_word_embedding_features)
#predictions = model.predict(test_word_embedding_features)

# casually printing results
#for sent, pred in zip(sents_train,predictions):
    #print(sent, pred, "\n")
print("Predictions:\n", predictions)

# Confusion matrix
test_list = test_labels_primary.tolist()
#dev_list = dev_labels_primary.tolist()
pred_list = [pred for pred in predictions]
labels=[1,3,5,4,2]
path='output/AI Classifier/1Label_confusion_matrix_NormTrue.png'
display_labels=['Commit to privacy', 'Declare opinion about privacy','Not applicable','Related to privacy','Violate privacy']
# NORMALIZED CONFUSION MATRIX
#create_confusion_matrix(dev_list, pred_list, "true", path, labels, display_labels)
create_confusion_matrix(test_list, pred_list, "true", path, labels, display_labels)
# NON NORMALIZED CONFUSION MATRIX
path='output/AI Classifier/1Label_confusion_matrix_NonNorm.png'
#create_confusion_matrix(dev_list, pred_list, None, path, labels, display_labels)
create_confusion_matrix(test_list, pred_list, None, path, labels, display_labels)

# FLAG - CHECK IF CONFUSION MATRIX IS CORRECT FOR EVERY LABEL
#path='output/AI Classifier/1labelPredictionsStatsDev.txt'
path='output/AI Classifier/1labelPredictionsStatsTest.txt'
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, 'w') as file:
    #print("Performance measures - Unigram Dictionary - MLP Word Embeddings\n", file=file)
    print("Performance measures - Unigram Dictionary - Adaboost\n", file=file)
#write_output_stats_file(path, "Dev", dev_labels_primary, predictions, labels)
write_output_stats_file(path, "Test", test_labels_primary, predictions, labels)

# TO DO: WRITE PREDICTIONS JSON FILE -> LEARN HOW TO TRANSFORM ADABOOST OUTPUT IN DICT ( LIST OF ({"text":sentence['text'], "label":label}))
#write_predictions_file("Dev", dev_pred_dict)
#write_predictions_file("Test", test_pred_dict)

# help reference: https://newbedev.com/valueerror-could-not-broadcast-input-array-from-shape-2242243-into-shape-224224
# https://blog.paperspace.com/adaboost-optimizer/
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit