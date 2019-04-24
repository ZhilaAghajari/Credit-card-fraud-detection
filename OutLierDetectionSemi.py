# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:25:58 2019

@author: Dai
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_curve, auc
#import scipy as sp
#from sklearn.utils.multiclass import unique_labels

from sklearn import svm
#from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def balance(data):
    sincere = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    sincere = sincere.sample(frac=1)
    fraudOverSample = fraud
    while len(sincere) - len(fraudOverSample) > 0:
        fraudOverSample = fraudOverSample.append(fraud, ignore_index=True)
#        
#    sincere = sincere[: len(fraud)]
    balanced_data = sincere.append(fraudOverSample, ignore_index=True)
    balanced_data = balanced_data.sample(frac=1)
    return balanced_data

#def scorer(pipeline, X, y):
#    y_score = - pipeline.decision_function(X)
#    score = roc_auc_score(y, y_score) * 100.
#    return score

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def plot_ROC(y_true, y_score):
    fpr, tpr, threshold = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    return roc_auc

data = pd.read_csv("creditcard.csv")

balanced_data = balance(data)

X = data.drop(["Time", "Amount"], axis=1)

#X = balanced_data.drop(["Time", "Amount"], axis=1)

X = X.as_matrix()
y = X[:, -1]

outliers_fraction = 0.002

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X[:, :-1], y, test_size=0.2)

#Elliptic Envelipe model
pipeline2 = EllipticEnvelope(support_fraction = 1, contamination=outliers_fraction)
print("=======================Elliptic Envelope Parameter====================")
print(pipeline2.get_params())

pipeline2.fit(X_train)

pred2 = -(pipeline2.predict(X_test) - 1)/2
pred2 = pred2.astype(int)
#score2 = scorer(pipeline2, X_test, y_test)
yscore2 = - pipeline2.decision_function(X_test)
print("========================Elliptic Envelope Performance====================")
print(classification_report(y_test, pred2, target_names=['Sincere', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred2, classes=['Sincere', 'Fraud'],title='Confusion matrix of Elliptic Envelope' )
print("ROC curve:")
score2 = plot_ROC(y_test, yscore2)
print("The AUC is:" + str(score2))
print("\n")



#Isolation forest model
print("========================Isolation Forest Parameters====================")
print(IsolationForest().get_params())

pipeline = IsolationForest(behaviour = 'new', contamination=outliers_fraction)

pipeline.fit(X_train)

pred = -(pipeline.predict(X_test) - 1)/2
pred = pred.astype(int)
#score = scorer(pipeline, X_test, y_test)
yscore = - pipeline.decision_function(X_test)
print("========================Isolation Forest Performance====================")
print(classification_report(y_test, pred, target_names=['Sincere', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred, classes=['Sincere', 'Fraud'],title='Confusion matrix of Isolation Forest' )
print("ROC curve:")
score = plot_ROC(y_test, yscore)
print("The AUC is:" + str(score))
print("\n")


#One class SVM model
pipeline3 = svm.OneClassSVM(gamma = 'auto', nu = 0.002)
print("=======================One-Class SVM Parameter====================")
print(pipeline3.get_params())

pipeline3.fit(X_train)
pred3 = -(pipeline3.predict(X_test) - 1)/2
pred3 = pred3.astype(int)
#score3 = scorer(pipeline3, X_test, y_test)
yscore3 = - pipeline3.decision_function(X_test)
print("========================One-Class SVM Performance====================")
print(classification_report(y_test, pred3, target_names=['Sincere', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred3, classes=['Sincere', 'Fraud'],title='Confusion matrix of One-Class SVM' )
print("ROC curve:")
score3 = plot_ROC(y_test, yscore3)
print("The AUC is:" + str(score3))
print("\n")




#LOF model
pipeline4 = LocalOutlierFactor(novelty = True, contamination=outliers_fraction, n_neighbors = 5)
print("=======================LocalOutlierFactor Parameter====================")
print(pipeline4.get_params())

pipeline4.fit(X_train)
pred4 = -(pipeline4.predict(X_test) - 1)/2
pred4 = pred4.astype(int)
#score4 = scorer(pipeline4, X_test, y_test)
yscore4 = - pipeline4.decision_function(X_test)
print("========================LocalOutlierFactor Performance====================")
print(classification_report(y_test, pred4, target_names=['Sincere', 'Fraud']))
print("Confusin Matix:")
plot_confusion_matrix(y_test, pred4, classes=['Sincere', 'Fraud'],title='LocalOutlierFactor' )
print("ROC curve:")
score4 = plot_ROC(y_test, yscore4)
print("The AUC is:" + str(score4))
print("\n")