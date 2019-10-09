# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:39:09 2019

@author: Maciek Berbeka
"""

#Preforming logistic reggression on data set about expeditions of Nepal Hymalayans in period of 1960-2017.
#Libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

exp = pd.read_excel("expeditions_data.xlsx")
data1 = pd.read_excel("year_y.xlsx")

#Plots

p1 = sns.countplot(x='year', data=exp,) #zmień etykiety żeby były pionowo (Wykres1)
p2 = sns.countplot(x='Y', data=exp ,palette='RdBu_r')
p3 = sns.countplot(x='year', data=exp,)
#Fuction showin succes rate


n_succes = np.count_nonzero(exp["Y"].values == "T")
n = exp["Y"].shape[0]
print("During all researhed years succes rate was: " +str((n_succes/ n)*100) +"%")

Y1 = exp["Y liczb"].values
X1 = exp.drop(["Y", "smtdays(korekta)","India", "o2climb", "o2sleep", "o2medical", "primrte", "primmem", "primref", "summer", "winter", "Y liczb", "highpoint", "rope^2", "ln(totdays)", "totmembers^2", "ln(totmembers)", "ln(rope)"], 1)

'''
#Coreelation check
plt.matshow(X1.corr(method = "kendall"))
plt.show()
'''
multicolinearity_check = X1.corr()

#Box-Tidwell test
linearity_check_df = pd.concat([X1,pd.DataFrame(Y1)],axis=1)


#sns.regplot(x= 'year', y= "Y liczb" , data= exp, logistic= True).set_title("Log Odds Linear Plot")
#sns.regplot(x= 'o2used', y= 'Y liczb', data= exp, logistic= True).set_title("Log Odds Linear Plot")
#sns.regplot(x= 'hratio', y= 'Y liczb', data= exp, logistic= True).set_title("Log Odds Linear Plot")




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


print("Spliting data to test set and training set:")
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X7, Y1, test_size=0.2, random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy: {:.2f}'.format(logreg.score(X_test, y_test)))

#Nie działa, moze inaczej typo zdefiniował, logreg
'''
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(logreg.grid_scores_) + 1), logreg.grid_scores_)
plt.show()
'''

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix")
print(confusion_matrix)

print("Accuracy:", str('\t {0:4.2f}'.format(metrics.accuracy_score(y_test, y_pred))))
print("Precision:", str('\t {0:4.2f}'.format(metrics.precision_score(y_test, y_pred))))
print("Recall:", str('\t {0:4.2f}'.format(metrics.recall_score(y_test, y_pred))))


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1],'r--')
plt.show()
