from scipy.sparse import data
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from partition import sents_train, labels_train, sents_test
import re
from sent2vec.vectorizer import Vectorizer
import spacy
import nltk

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
corpus = []
words_to_numbers = {}
number_representation = 0
vectors_list = []

for sent in sents_train:
    sent_doc = nlp(sent)
    sent_tokens_list = []
    sent_vector = []
    for token in sent_doc:
        sent_tokens_list.append(token.text)
        if token.text not in words_to_numbers:
            words_to_numbers[token.text] = number_representation
            number_representation += 1
        sent_vector.append(words_to_numbers[token.text])
    dataset_tokens.append(sent_tokens_list)
    vectors_list.append(sent_vector)
    corpus.extend(sent_tokens_list)

freq = nltk.FreqDist(corpus)

print(vectors_list)
print(dataset_tokens)
print(number_representation)
print (freq)

#print(labels_train, type(labels_train))
#vectorizer = Vectorizer()
#vectorizer.bert(sents_train)
#sent_vectors = vectorizer.vectors
# Reshaping needed... create word embeddings
#sents_train = sents_train.reshape(-1,1)
#print(sent_vectors)
# Configurations
#adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training
#model = adaclassifier.fit(sents_train, labels_train)


# Predicting
#predictions = model.predict(sents_test)

#print(predictions)

# Measuring results
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED TO OUTLIERS - declare opinion about privacy is a very rare category...