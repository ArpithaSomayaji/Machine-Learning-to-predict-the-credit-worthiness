import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn

from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report


#Read csv into DataFrame
creditData=pd.read_csv('creditData.csv')

#Print first 5 rows and columns in the dataframe 
print(creditData.head())
list(creditData.columns.values)

#Checking for missing values
creditData.isnull().sum()


#Find which datatypes are present in dataframe
creditData.dtypes

#Find the data Types and their Frequency  
print("Data types and their frequency\n{}".format(creditData.dtypes.value_counts()))

# Let’s select just the object columns using the DataFrame method select_dtype, then display a sample row to get a better sense of how the values in each column are formatted.
object_columns_df = creditData.select_dtypes(include=['object'])
print(object_columns_df.iloc[0])


#Find number of unique values in each feature
for name in creditData.columns:
    print(name,':')
    print(creditData[name].value_counts(),'\n')
	
	
	
#Eliminate Columns 	that have too many Unique Values 
drop_cols = ['duration','credit_amount','age']
filtered_creditData={}
filtered_creditData =creditData.drop(drop_cols,axis=1)

#Change all Ordinal Feature values into Numeric
mapping_dict = {
    "checking_status":{
        "'no checking'" : 0,
        "<0":1,
        "0<=X<200":2,
        ">=200":3 
    },
    "credit_history" : {
        "'critical/other existing credit'":0,
        "'delayed previously'":1,
        "'existing paid'":2,
        "'no credits/all paid'":3,
        "'all paid'":4 
    },
    "savings_status":{
        "'no known savings'":0,
        "<100":1,
        "100<=X<500":2,
        "500<=X<1000":3,
        ">=1000":4             
    },
    "employment":{
        "unemployed":0,
        "<1":1,
        "1<=X<4":2,
        "4<=X<7":3,
        ">=7":4        
    },
    "job":{
        "'unemp/unskilled non res'":0,
        "'unskilled resident'":1,
        "skilled":2,
        "'high qualif/self emp/mgmt'":3      
        
    },
    "class":{
        "good":1,
        "bad":0
    }
    
    
}


updated_creditData=filtered_creditData.replace(mapping_dict)
updated_creditData[['checking_status','credit_history','savings_status','employment','job','class']].head()


#approach to converting nominal features into numerical features is to encode them as dummy variables
nominal_columns = ["purpose", "installment_commitment", "personal_status", "other_parties","residence_since","property_magnitude","other_payment_plans","housing","existing_credits","num_dependents","own_telephone","foreign_worker"]
dummy_df = pd.get_dummies(updated_creditData[nominal_columns])
updated_creditData_v2={}
final_updated_creditData={}
updated_creditData_v2 = pd.concat([updated_creditData, dummy_df], axis=1)
final_updated_creditData = updated_creditData_v2.drop(nominal_columns, axis=1)
final_updated_creditData.head()


#Cross-Verify the Datatypes of all Features
final_updated_creditData.info()

# Save the cleaned Data as CSV

final_updated_creditData.to_csv("cleaned_creditData.csv",index=False)

# Source : https://www.dataquest.io/blog/machine-learning-preparing-data/

#Spilt DATA to X and y
X = final_updated_creditData.iloc[:,final_updated_creditData.columns!='class']
y = final_updated_creditData.iloc[:,final_updated_creditData.columns=='class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)


#-----------------------------------------------------------------------------------------------------------------------
#Logistic Regression Implementation 
LogReg = LogisticRegression()
LogReg.fit(X_train, y_train)

y_pred = LogReg.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

print(classification_report(y_test, y_pred))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(LogReg.score(X_test, y_test)))


# store the predicted probabilities for class 1
y_pred_prob = LogReg.predict_proba(X_test)[:, 1]

print(metrics.roc_auc_score(y_test, y_pred_prob))

#Source : http://www.ritchieng.com/machine-learning-evaluate-classification-model/


#--------------------------------------------------------------------------------------------------------------------------
#Linear SVC Implementation

from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix

print(classification_report(y_test, y_pred))


print('Accuracy of logistic SVC classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

y_score = clf.decision_function(X).ravel()
print (roc_auc_score(y, y_score))

print(clf.coef_)
print(clf.intercept_)

#--------------------------------------------------------------------------------------------------------------------------
#Decision Tree Implementation

from sklearn import tree

treeclf = tree.DecisionTreeClassifier(max_depth=3)
treeclf.fit(X_train,y_train)
y_pred = treeclf.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(treeclf.score(X_test, y_test)))

# store the predicted probabilities for class 1
y_pred_prob = treeclf.predict_proba(X_test)[:, 1]
print(metrics.roc_auc_score(y_test, y_pred_prob))

#--------------------------------------------------------------------------------------------------------------------------
#random Forest Implementation - with preprossing 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

RFClf = RandomForestClassifier(max_depth=2, random_state=0)
RFClf.fit(X_train,y_train)
y_pred = RFClf.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RFClf.score(X_test, y_test)))
y_pred_prob = RFClf.predict_proba(X_test)[:, 1]
print(metrics.roc_auc_score(y_test, y_pred_prob))


importances = RFClf.feature_importances_
std = np.std([tree.feature_importances_ for tree in RFClf.estimators_],axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, str(X_train.columns[f]), importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X_train.columns)
plt.xlim([-1, X_train.shape[1]])
plt.show()
#--------------------------------------------------------------------------------------------------------------------------
#Multi-Layered Perceptron (Neural Network) Implementation

from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier()
MLP.fit(X_train,y_train)
y_pred = MLP.predict(X_test)
print(classification_report(y_test, y_pred))
print('Accuracy of logistic Multi Layered on test set: {:.2f}'.format(MLP.score(X_test, y_test)))
# store the predicted probabilities for class 1
y_pred_prob = MLP.predict_proba(X_test)[:, 1]
print(metrics.roc_auc_score(y_test, y_pred_prob))

#--------------------------------------------------------------------------------------------------------------------------
#random Forest Implementation - without preprocessing 

creditData=pd.read_csv('creditData.csv')

drop_cols = ['duration','credit_amount','age']
creditData = creditData.drop(drop_cols,axis=1)
mappedData = creditData
mapped_values={}

for col in creditData:
    a=0;
    if pd.api.types.is_object_dtype(creditData[col]):
        print (col, creditData[col].unique())
        for colvalue in creditData[col].unique():
            mapped_values[colvalue] = a
            a = a + 1;
        mappedData = mappedData.replace(mapped_values)

X = mappedData.iloc[:,mappedData.columns!='class']
y = mappedData.iloc[:,mappedData.columns=='class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)
        

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, str(X_train.columns[f]), importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), X_train.columns)
plt.xlim([-1, X_train.shape[1]])
plt.show()