
# coding: utf-8

# In[67]:


# coding: utf-8

# # Spam classification

# ## Import libraries

# In[5]:
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import time
import collections
import re
import random
import scipy.io
import glob
from sklearn.model_selection import train_test_split
from pandas_ml import ConfusionMatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from nltk import PorterStemmer


# In[6]:


#setting the path for the location
import os
os.chdir('D:/Freelancer_questions/Spam_ham/Enron-Email-Classification-master/Enron-Email-Classification-master')


# In[11]:


# ## Vectorizer
#reading the input data from the txt spam vs ham file
# In[8]:

NUM_TRAINING_EXAMPLES = 5172
NUM_TEST_EXAMPLES = 5857

BASE_DIR = './'
SPAM_DIR = 'spam/'
HAM_DIR = 'ham/'
TEST_DIR = 'test/'

vectorizer = TfidfVectorizer(input='filename',lowercase=True, stop_words="english",
                             encoding='latin-1',min_df=8) 

spam_filenames = glob.glob( BASE_DIR + SPAM_DIR + '*.txt')
print(spam_filenames)
ham_filenames = glob.glob( BASE_DIR + HAM_DIR + '*.txt')
test_filenames = [BASE_DIR + TEST_DIR + str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]
all_filenames = spam_filenames + ham_filenames # including test_filenames


# In[13]:


#converted the data to countvectorizer and created a dictionary on the same#
train_matrix = vectorizer.fit_transform(all_filenames)
test_matrix = vectorizer.transform(test_filenames)
X = train_matrix
Y = [1]*len(spam_filenames) + [0]*len(ham_filenames)

# Save as .mat 
file_dict = {}
file_dict['training_data'] = X
file_dict['training_labels'] = Y
file_dict['test_data'] = test_matrix
scipy.io.savemat('email_data.mat', file_dict)


# In[21]:


#checking the dictionary sample
file_dict['training_data'].todense()


# In[50]:
data = scipy.io.loadmat('./email_data.mat')

train_X = data['training_data'].toarray()
train_y = data['training_labels'].reshape(X.shape[0],1)
X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
test_X = data['test_data'].toarray()
dt = DecisionTreeClassifier() 
    
clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt,learning_rate=1).fit(X_train, y_train)
predicted_test = clf.predict(test_X)
predicted_val = clf.predict(X_val)


# In[68]:


data = scipy.io.loadmat('./email_data.mat')
def adaboost_submission(data):
    # combine test and training data for scaling
    
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    test_X = data['test_data'].toarray()
    dt = DecisionTreeClassifier() 
    
    clf = AdaBoostClassifier(n_estimators=50, base_estimator=dt,learning_rate=1).fit(X_train, y_train)
    predicted_test = clf.predict(test_X)
    predicted_val = clf.predict(X_val)
 

    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_ada = adaboost_submission(data)


# In[69]:


def lda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = LinearDiscriminantAnalysis().fit(X_train, y_train)
    predicted_test = clf.predict(test_X)
    predicted_val = clf.predict(X_val)
 

    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_lda = lda_submission(data)


# In[70]:


def qda_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
    predicted_test = clf.predict(test_X)
    predicted_val = clf.predict(X_val)
 

    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_qda = qda_submission(data)


# In[71]:


def xgboost_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = XGBClassifier( max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8).fit(X_train, y_train)
    predicted_test = clf.predict(test_X)
    predicted_val = clf.predict(X_val)
 

    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_xgm = xgboost_submission(data)


# In[72]:


def randomforest_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    
    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_rf = randomforest_submission(data)


# ### Logistic Regression

# In[13]:

def logreg_submission(data):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = LogisticRegression(C=1).fit(X_train, y_train)
    
    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test

submit_log = logreg_submission(data)


# In[73]:


def svm_submission(data, c):
    # combine test and training data for scaling
    train_X = data['training_data'].toarray()
    train_y = data['training_labels'].reshape(X.shape[0],1)
    test_X = data['test_data'].toarray()
    X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.20, random_state=42)
    clf = LinearSVC(C=c).fit(X_train, y_train)
    
    print ('Accuracy:', accuracy_score(y_val, predicted_val))
    print ('F1 score:', f1_score(y_val, predicted_val))
    print ('Recall:', recall_score(y_val, predicted_val))
    print ('Precision:', precision_score(y_val, predicted_val))
    print ('\n clasification report:\n', classification_report(y_val, predicted_val))
    print ('\n confussion matrix:\n',confusion_matrix(y_val, predicted_val))
    
    return predicted_test
submit_svm = svm_submission(data, 0.1)


# In[74]:


#vote of count method for output ensemble
from scipy import stats
submit = [submit_log[i]+submit_rf[i]+submit_xgm[i]+submit_svm[i]+submit_ada[i] for i in range(len(submit_svm))]
submit = np.asarray(submit)

submit[np.where(submit==1)] = 0
submit[np.where(submit==2)] = 0
submit[np.where(submit==3)] = submit_log[np.where(submit==3)]
submit[submit > 3] = 1




# In[43]:


# ### Save as .csv

# In[ ]:

df = pd.DataFrame(submit)
df.index += 1
df['Id'] = df.index
df.columns = ['Category', 'Id']
df.to_csv("submit_new.csv",header=True,columns=['Id','Category'],index = False)

