# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:39:09 2019

@author: Maciek Berbeka
"""

#Performing logistic reggression on data set about expeditions in Nepal Hymalayans in the period of 1960-2017.
#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import succes_rate_function

#Reading data 
exp = pd.read_excel("expeditions_data.xlsx")
data1 = pd.read_excel("year_y.xlsx")

#Plots
#Vizualization of number of expeditions during the researched period
plt.figure()
succes_rate_function.succes_rate(data1, 1950, 2017 , 1, "count")
n_succes = np.count_nonzero(exp["Y"].values == "T")
n = exp["Y"].shape[0]
print("During all researhed years success rate was: " +str((n_succes/ n)*100) +"%")

#Function showing success rate, returning plot. First parameter is dataset, do not modyfy
#Second and third parameter are years showing first and last year wchih will be analyzed
#Minimal value -1950, max 2017, those are also implide values
#Last parameter is interval,  please keep it as integer

succes_rate_function.succes_rate(data1, 1980, 2010 , 5)

#Defining Y and X variables for logisitc regression model
Y1 = exp["Y liczb"].values
X1 = exp.drop(["Y", "smtdays(korekta)","India", "o2climb", "o2sleep", "o2medical", "primrte", "primmem", "primref", "summer", "winter", "Y liczb", "highpoint", "rope^2", "ln(totdays)", "totmembers^2", "ln(totmembers)", "ln(rope)"], 1)

#Correlation check
plt.matshow(X1.corr(method = "kendall"))
plt.show()

multicolinearity_check = X1.corr()
print(multicolinearity_check)


#Box-Tidwell test

sns.regplot(x= 'year', y= "Y liczb" , data= exp, logistic= True).set_title("Log Odds Linear Plot")

#Performing logistic regression. After the first iteration variables with the lowest statistical significance were remowed
#Then another estimation was made. Process was repeated untill all variables were stastically signifiacant
import statsmodels.api as sm
print("logistic reggresion model number 1")
logit1 = sm.Logit(Y1,X1)
result = logit1.fit()
print(result.summary())

print("logistic reggresion model number 2")
X2 = X1.drop(["stdrte"], 1)
logit2 = sm.Logit(Y1,X2)
result = logit2.fit()
print(result.summary())

print("logistic reggresion model number 3")
X3 = X2.drop(["totdays"], 1)
logit3 = sm.Logit(Y1,X3)
result = logit3.fit()
print(result.summary())

print("logistic reggresion model number 4")
X4 = X3.drop(["China"], 1)
logit4 = sm.Logit(Y1,X4)
result = logit4.fit()
print(result.summary())

print("logistic reggresion model number 5")
X5 = X4.drop(["fall"], 1)
logit5 = sm.Logit(Y1,X5)
result = logit5.fit()
print(result.summary())

print("logistic reggresion model number 6")
X6 = X5.drop(["totmembers"], 1)
logit6 = sm.Logit(Y1,X6)
result = logit6.fit()
print(result.summary())

print("logistic reggresion model number 7")
X7 = X6.drop(["rope"], 1)
logit7 = sm.Logit(Y1,X7)
result = logit7.fit()
print(result.summary())


#Splitting data to test set and training set
print("Spliting data to test set and training set:")
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE


#Calculating accuracy of model number 7
classifier = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X7, Y1, test_size=0.3, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print('Accuracy: {:.2f}'.format(classifier.score(X_test, y_test)))


# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2), scoring='accuracy')
rfecv.fit(X_train, y_train)
print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)

from sklearn.metrics import classification_report
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
print(classification_report(y_test, y_pred))

print("Accuracy:", str('\t {0:4.2f}'.format(metrics.accuracy_score(y_test, y_pred))))
print("Precision:", str('\t {0:4.2f}'.format(metrics.precision_score(y_test, y_pred))))
print("Recall:", str('\t {0:4.2f}'.format(metrics.recall_score(y_test, y_pred))))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.show()
