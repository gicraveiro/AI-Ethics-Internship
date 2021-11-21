from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from partition import sents_train, labels_train, sents_test
import re
from sent2vec.vectorizer import Vectorizer

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

#print(labels_train, type(labels_train))
#vectorizer = Vectorizer()
#vectorizer.bert(sents_train)
#sent_vectors = vectorizer.vectors
# Reshaping needed... create word embeddings
#sents_train = sents_train.reshape(-1,1)
#print(sent_vectors)
# Configurations
adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

# Training
#model = adaclassifier.fit(sent_vectors, labels_train)


# Predicting
#predictions = model.predict(sents_test)

#print(predictions)

# Measuring results
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# CAREFUL
# ADABOOST IS HIGHLY AFFECTED TO OUTLIERS - declare opinion about privacy is a very rare category...