#***********************************************************************************************
# Title:                 XXXXXX 
# Author:                Ana Margarida Silva da Costa & Lea Katharina Paoli
# Matriculation Nr.      XXXXXX & E12499
# Purpose:               XXXXXX 
#************************************************************************************************



#***********************IMPORT MODULES***********************

# Installation of packages [Use the command line, not the Python shell]
"""
pip install numpy
pip install --upgrade pandas-datareader
pip install -U scikit-learn

for help, see: https://phoenixnap.com/kb/install-pip-windows
"""

# Import the Python modules you need
import numpy as np
import pandas as pd
from sklearn.model_selectionction import train_test_split
from sklearn.linear_model import LogisticRegression



#***********************PREPATORY WORK***********************
#Load the dataset as a Pandas Data Frame
df=pd.read_excel("JSTdatasetR4.xlsx",sheet_name="Data")

#Creating the desired variables in a new Data Frame
#let's make a copy, in order to preserve the original dataset
df_copy=df.copy()

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
return (x - lag) / lag


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
#look at dataset to get an impression of its structure first -> df.shape will return (x1, x2), where x1=nrrows and x2=nrcolumns
df.shape

#DELETE: https://www.youtube.com/watch?v=fwY9Qv96DJY
#train_test_split(df[['X {= something multidimensional -> 2 brackets}']], df.Y{=explanatory variable}, test_size={percentage of the sample used for testing, e.g. 0,1})
X_train, X_test, y_train, y_test = train_test_split()
#results we get back from train_test_split(): X_train, X_test, y_train, y_test
#df

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


#--- MODEL2: Logistic Regression with LASSO Regularization.
# regularization parameter 
# using a 5-fold cross validation

# Which variables survive? (Exercise 5)



#--- MODEL3: Random Trees
# Experiment with different tree depths, not necessarily with a cross validation



#--- MODEL4: random forest
## MODEL5: neural networks.
# Experiment with different numbers of hidden layers, and neurons for each layers, not necessarily using a cross-validation

#-------- ROC curves
#Plot the ROC curves for the best versions of your models and compute the AUROC. According to this
# criterion, which model performs best ?

#-------- Confusion Matrices Comparison
