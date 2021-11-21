from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from partition import sents_train, labels_train, sents_test
from sklearn import datasets
import re

#print(sents_train, type(sents_train))
for row_id,row in enumerate(sents_train):
    row = re.sub("\n", " ", row)
    sents_train[row_id] = row
for row_id,row in enumerate(sents_test):
    row = re.sub("\n", " ", row)
    sents_test[row_id] = row

#print(sents_train, type(sents_train))
#print(type(iris))
#print(type(X))
#print(type(y))

adaclassifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)


#print(partition.sents_train)
# Training
model = adaclassifier.fit(sents_train, labels_train)

# Predict

#predictions = model.predict(sents_test)

#print(predictions)