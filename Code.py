#DELETE: Add info in title in the end!
#***********************************************************************************************
# Title:                 XXXXXX 
# Author:                Ana Margarida Silva da Costa & Lea Katharina Paoli
# Matriculation Nr.      XXXXXX & XXXXX
# Purpose:               XXXXXX 
#************************************************************************************************


#***********************IMPORT MODULES***********************

# Installation of packages [Use the command line, not the Python shell]
"""
py -m pip install numpy
py -m pip install --upgrade pandas-datareader
py -m pip install -U scikit-learn
pip install openpyxl
pip install seaborn

#check whether the installation worked properly: 
py -m pip list
for help, see: https://phoenixnap.com/kb/install-pip-windows & https://www.youtube.com/watch?v=SrX5yo4KKGM & https://www.youtube.com/watch?v=RvbUqf3Tb1s
"""

# Import the Python modules you need
import numpy as np
import pandas as pd
import pickle # allows us to store training date into a file 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold #as opposed to KFold 

import matplotlib.pyplot as plt
import seaborn as sn 

#***********************PARAMETRIZATION***********************
print("***********************PARAMETRIZATION***********************")
# Percentage test dataset 
percent_test = 0.2
print("percent_test is set to " + str(percent_test))

# Input Data
path_data = r"C:\Users\leapa\PycharmProjects\AdvFinEcon_PythonAssignment\JSTdatasetR4.xlsx"
print("Your excel file should be stored here: " + path_data)

# Cross-Validation Splits
number_of_splits = 5
kf = StratifiedKFold(n_splits=number_of_splits)
print("number_of_splits is set to " + str(number_of_splits))



#***********************PREPATORY WORK***********************
print("***********************PREPATORY WORK***********************")
#Load the dataset as a Pandas Data Frame
df=pd.read_excel(path_data ,sheet_name="Data", engine='openpyxl') #engine had to be added to open xlxs file


#Creating the desired variables in a new Data Frame
#let's make a copy, in order to preserve the original dataset
df_copy=df.copy()
print("this is the original:") #added
print(df.head()) #added
print("this is the copy:") #added
df_copy.head() #added

#let's create new (temporary) columns with the transformed variables we need:
#-slope of the yield curve
df_copy["slope_yield_curve"]=df_copy["ltrate"]/100-df_copy["stir"]/100

# credit: loans to the privete sector / gdp
df_copy["credit"]=df_copy["tloans"]/df_copy["gdp"]

# debt service ratio: credit * long term interest rate
df_copy["debt_serv_ratio"]= (df_copy["tloans"]/df_copy["gdp"])*df_copy["ltrate"]/100

# broad money over gdp
df_copy["bmoney_gdp"]=df_copy["money"]/df_copy["gdp"]

# current account over gdp
df_copy["curr_acc_gdp"]=df_copy["ca"]/df_copy["gdp"]


# Now we need to compute 1-year absolute variations and percentage
# variations for a few variables
# Obviously this must be done country-wise, so we cannot act on the
# dataframe as it is.
# a Convenient way of doing this is the Pandas method 'groupby()'
df_copy_group=df_copy.groupby("iso") # 'iso' is the country code

# create 1 year-variation of credit from grouped dataframe and add
# back to initial dataframe
df_copy["delta_credit"]=df_copy_group["credit"].diff(periods=1)

# create 1 year-variation of debt ser ratio from grouped dataframe
# and add back to initial dataframe
df_copy["delta_debt_serv_ratio"]=df_copy_group["debt_serv_ratio"].diff(periods=1)

# create 1 year-variation of investment/gdp from grouped dataframe
# and add back to initial dataframe
df_copy["delta_investm_ratio"]=df_copy_group["iy"].diff(periods=1)

# create 1 year-variation of public debt/gdp from grouped dataframe
# and add back to initial dataframe
df_copy["delta_pdebt_ratio"]=df_copy_group["debtgdp"].diff(periods=1)

# create 1 year-variation of broad money / gdp from grouped dataframe
# and add back to initial dataframe
df_copy["delta_bmoney_gdp"]=df_copy_group["bmoney_gdp"].diff(periods=1)

# create 1 year-variation of current / gdp from grouped dataframe and
# add back to initial dataframe
df_copy["delta_curr_acc_gdp"]=df_copy_group["curr_acc_gdp"].diff(periods=1)

# now we need to create new variables which are 1-year growth rates
# of existing ones
# we will need this function to apply to the columns of the dataframe
def lag_pct_change(x):""" Computes percentage changes """
lag = np.array(pd.Series(x).shift(1))
return ((x - lag) / lag) #brackets added


# create 1 year growth rate of CPI from grouped dataframe and add
# back to initial dataframe
df_copy["growth_cpi"]=df_copy_group["cpi"].apply(lag_pct_change)

# create 1 year growth rate of consumption per capita from grouped
# dataframe and add back to initial dataframe
df_copy["growth_cons"]=df_copy_group["rconpc"].apply(lag_pct_change)

# low let's create the crises early warning label: a dummy variable
# which takes value one if in the next year or two there will be a crises
# temporary array of zeros, dimension = number of rows in database
temp_array=np.zeros(len(df_copy))

# loop to create dummy
for i in np.arange(0,len(df_copy)-2):
    temp_array[i]= 1 \
    if ( (df_copy.loc[i+1,'crisisJST']== 1) or (df_copy.loc[i+2,'crisisJST']== 1) ) else 0

#put the dummy in the dataframe
df_copy["crisis_warning"]=temp_array.astype("int64")
# create a smaller dataframe including only the variables we are
# interested in: the first ten are predictors (X) and the last one is
#the output, or label (y). Also, drop the observations where some
# variable is missing
variables=["slope_yield_curve","delta_credit","delta_debt_serv_ratio","delta_investm_ratio","delta_pdebt_ratio","delta_bmoney_gdp","delta_curr_acc_gdp","growth_cpi","growth_cons","eq_tr","crisis_warning"]
df_final=df_copy[variables].dropna()
# let's also create a version of our dataframe which includes the year
df_final_withyear=df_copy[["year"]+variables].dropna()

#***********************ANALYSIS***********************

#-------- Split Sample
# Training vs. Test (randomly assigned)
# look at dataset to get an impression of its structure first -> df.shape will return (x1, x2), where x1=nrrows and x2=nrcolumns
df.shape

#DELETE: https://www.youtube.com/watch?v=fwY9Qv96DJY
#train_test_split(df[['X {= something multidimensional -> 2 brackets}']], df.Y{=explanatory variable}, test_size={percentage of the sample used for testing, e.g. 0,1})
# add "random_state=" to keep sample allocation fixed when the code is executed multiple times
X_train, X_test, y_train, y_test = train_test_split(test_size=)
#results we get back from train_test_split(): X_train, X_test, y_train, y_test



# Varify 
N=len(x)
N
len(X_train) #should equal to percent_test*N
len(x_test) #should equal to (1-percent_test)*N

#-------- Fit Model
#--- MODEL1: logistic regression

# NOTES:
# Logistic regression predicts whether something is true or false -> discrete
# fits a logistic function to the data -> indicates the probability of Y being true
# = Classification technique based on MLE
# DELETE: https://www.youtube.com/watch?v=zM4VZR0px8E

#create object of the class "LogisticRegression", which we call model1
model1= LogisticRegression()
#Train the model
model.fit(x_train, y_train)
#Prediction -> returns an array of 0s and 1s, corresponding to yes/no in the order of the data returned after executing "X_test"
model.predict(x_test)
# -> returns an matrix corresponding to probability of being [false, true] for corresponding data point
model.predict_proba(x_test)
# Accuracy of the model [between 0 and 1]
model.score(x_test, y_test)

#Pickle (alternative method would be joblib from sklearn)
with open('model_pickle', 'wb') as f: 
    pickle.dump(model, f) #dump model into file 

with open('model_pickle', 'rb') as f: 
    mp = pickle.load(f) #mp=object

#--- MODEL2: Logistic Regression with LASSO Regularization.
# regularization parameter 
# using a 5-fold cross validation

# Which variables survive? (Exercise 5)



#--- MODEL3: Random Trees
# Experiment with different tree depths, not necessarily with a cross validation
# DELETE: https://www.youtube.com/watch?v=PHxYNGo8NcI


#--- MODEL4: random forest
## MODEL5: neural networks.
# Experiment with different numbers of hidden layers, and neurons for each layers, not necessarily using a cross-validation
# DELETE: https://www.youtube.com/watch?v=ok2s1vV9XW0

model4=RandomForestClassifier()
model.fit(X_train, y_train)
model.score(X_test, y_test) #n_estimator shows number of trees used 
#-> can be modified by adding n_estimators= 
model4=RandomForestClassifier(n_estimators)
model.score(X_test, y_test)

#-------- ROC curves
#Plot the ROC curves for the best versions of your models and compute the AUROC. 
#DELETE: https://www.youtube.com/watch?v=gJo0uNL-5Qw

# Cross-Validation
from train_index, test_index in kf.split(ENTER NAME OF DATASET)
print(train_index, test_index)

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(x_test)

get_score(LogisticRegression, X_train, X_test, y_train, y_test)

for train_index, test_index in kf.split(df_final):
    X_train, X_test, y_train, y_test = df_final[train_index], df_final[test_index], df_final[train]  
#-------- Confusion Matrices Comparison
#Plot the true value of y against the prediced value 
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

%matplotlib inline
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('True Value')

#According to this criterion, which model performs best?


#NOTE: Bias (train error -> high error in training dataset = UNDERFIT) 
#vs Variance (test error -> High Variance = Accuracy of the model varies sustantially with the choice of test/train dataset =OVERFIT)
# low variance & low bias = BAlANCED FIT
# -> Cross Validation, Regularization, Dimensionality Reduction & Ensemble Techniques