#!/usr/bin/env python
# coding: utf-8

# # An Exercise of Financial Crises Prediction Using Machine Learning Techniques
# ## Inspired by Bluwstein et al. (2019)
# # Authors:                Ana Margarida Silva da Costa & Lea Katharina Paoli
# # Matriculation Nr.      Q00087 & E12499
# # Purpose:               Assignment for Advanced Financial Economics 

# The small scale reproduction of Bluwstein et a. (2019) constitutes a binary classification problem, where each datapoint characterized by a vector of predictors realized at time must be classified into one of two categories: 
# 1) there will be a financial crises at time t+1 or t+2.
# 
# 2) there will not be a financial crises at time t+1 or t+2. 
# where t is measured in years. 
# Time t is measured in years. 
# 
# Using five different machine learning techniques, the likelihood of a crisis occuring is to be assessed.  

# # Installation of necessary packages

# In[1]:


pip install numpy


# In[2]:


pip install pandas 


# In[3]:


pip install --upgrade pandas-datareader


# In[4]:


pip install plotly 


# In[5]:


pip install -U scikit-learn


# In[6]:


pip install openpyxl


# In[7]:


pip install matplotlib 


# In[8]:


pip install seaborn


# In[9]:


pip install qeds 


# Now to check whether the installation worked properly:

# In[10]:


pip list


# It appears as it has worked properly, since we can find the previous packages in the list. 

# # Importing the necessary Python modules

# In[11]:


import numpy as np
import pandas as pd 
import plotly.graph_objects as go

from sklearn import (
    model_selection, linear_model, ensemble, metrics, neural_network, pipeline, model_selection, \
    tree, preprocessing, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold #as opposed to KFold
import matplotlib.pyplot as plt
import seaborn as sn 

import itertools

import statsmodels.api as sm

import qeds 
qeds.themes.mpl_style();
plotly_template = qeds.themes.plotly_template()
colors = qeds.themes.COLOR_CYCLE


# # Parametrization
# Instead of hard-coding the percentage of the dataset used as a test-dataset as well as the number of cross-validation splits we chose to store them as variables. This way, these parameters can be changed easily when needed without being forced to make adjustment all across the file. 

# In[12]:


# Percentage test dataset 
percent_test = 0.25
print("percent_test is set to " + str(percent_test))

# Cross-Validation Splits
number_of_splits = 5
kf = StratifiedKFold(n_splits=number_of_splits)
print("number_of_splits is set to " + str(number_of_splits))


# # Preparatory work

# Loading the dataset:

# In[13]:


df=pd.read_excel("JSTdatasetR4.xlsx",sheet_name="Data")


# Creating a copy to preserve the original version of the dataset

# In[14]:


df_copy=df.copy()


# ### Creation of additional columns 

# slope of the yield curve

# In[15]:


df_copy["slope_yield_curve"]=df_copy["ltrate"]/100-df_copy["stir"]/100


# credit: loans to the private sector/gdp

# In[16]:


df_copy["credit"]=df_copy["tloans"]/df_copy["gdp"]


# debt service ratio: credit * long term interest rate

# In[17]:


df_copy["debt_serv_ratio"]=(df_copy["tloans"]/df_copy["gdp"])*df_copy["ltrate"]/100


# broad money over gdp

# In[18]:


df_copy["bmoney_gdp"]=df_copy["money"]/df_copy["gdp"]


# current account over gdp

# In[19]:


df_copy["curr_acc_gdp"]=df_copy["ca"]/df_copy["gdp"]


# ### Computatation of 1-year absolute variations and percentage variations 

# Computatation of 1-year absolute variations and percentage variations for
# - credit from grouped dataframe and add back to initial dataframenp.zeros
# - debt ser ratio from grouped dataframe and add back to initial dataframe
# - investment/gdp from grouped dataframe and add back to initial dataframe
# - public debt/gdp from grouped dataframe and add back to initial dataframe
# - broad money / gdp from grouped dataframe and add back to initial dataframe
# - current / gdp from grouped dataframe and add back to initial dataframe
# 
# (beforehand, a copy of the respective dataframes is being created)

# In[20]:


df_copy["delta_credit"]=np.empty(len(df_copy))*np.nan
df_copy["delta_debt_serv_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_investm_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_pdebt_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_bmoney_gdp"]=np.empty(len(df_copy))*np.nan
df_copy["delta_curr_acc_gdp"]=np.empty(len(df_copy))*np.nan
df_copy["growth_cpi"]=np.empty(len(df_copy))*np.nan
df_copy["growth_cons"]=np.empty(len(df_copy))*np.nan


# <font color='red'>IMPLEMENTATION OF REMARK 2</font>
# 
# Bluwstein et al (2019) clearly highlight the importance of the global slope of the yield curve and the global credit variation as financial crises predictors. 

# In[21]:


# first create a temporary column including the sum of GDP for each country grouped by year of observation
# check if the column name is actually called year

#df_copy["sum_gdp_by_year"]=df_copy.groupby("year")["gdp"].transform('sum')


# now create a new temporary column with the slope of the yield curve for each observation multiplied
# by the gdp weight


#df_copy["weighted_slope"]=df_copy["slope_yield_curve"]*df_copy["gdp"]/df_copy["sum_gdp_by_year"]


# finally create the column "global_slope_yield_curve" by summing the weighted domestic slopes,
# after grouping observation by year


#df_copy["global_slope_yield_curve"]=df_copy.groupby("year")["weighted_slope"].transform('sum')


# <font color='red'> repeat the same procedure for the variable *"delta credit"*</font>

# In[22]:


# first create a temporary column including the sum of GDP for each country grouped by year of observation
# check if the column name is actually called year

#df_copy["sum_gdp_by_year"]=df_copy.groupby("year")["gdp"].transform('sum')


# now create a new temporary column with the slope of the yield curve for each observation multiplied
# by the gdp weight

#df_copy["weighted_slope"]=df_copy["slope_yield_curve"]*df_copy["gdp"]/df_copy["sum_gdp_by_year"]


# finally create the column "global_slope_yield_curve" by summing the weighted domestic slopes,
# after grouping observation by year


#df_copy["global_slope_yield_curve"]=df_copy.groupby("year")["weighted_slope"].transform('sum')


# <font color='red'>last 2 rows need to be adjusted </font>

# In[23]:


for i in np.arange(1,len(df_copy)):
    if (df_copy.loc[i,'iso'] == df_copy.loc[i-1,'iso']):
        df_copy.loc[i,"delta_credit"]= df_copy.loc[i,'credit']-df_copy.loc[i-1,'credit']
        df_copy.loc[i,"delta_debt_serv_ratio"]= df_copy.loc[i,'debt_serv_ratio']-df_copy.loc[i-1,'debt_serv_ratio']
        df_copy.loc[i,"delta_investm_ratio"]= df_copy.loc[i,'iy']-df_copy.loc[i-1,'iy']
        df_copy.loc[i,"delta_pdebt_ratio"]= df_copy.loc[i,'debtgdp']-df_copy.loc[i-1,'debtgdp']
        df_copy.loc[i,"delta_bmoney_gdp"]= df_copy.loc[i,'bmoney_gdp']-df_copy.loc[i-1,'bmoney_gdp']
        df_copy.loc[i,"delta_curr_acc_gdp"]= df_copy.loc[i,'curr_acc_gdp']-df_copy.loc[i-1,'curr_acc_gdp']
        df_copy.loc[i,"growth_cpi"]= (df_copy.loc[i,'cpi']-df_copy.loc[i-1,'cpi'])/df_copy.loc[i-1,'cpi']
        df_copy.loc[i,"growth_cons"]= (df_copy.loc[i,'rconpc']-df_copy.loc[i-1,'rconpc'])/df_copy.loc[i-1,'rconpc']
        
        #df_copy.loc[i,"global_slope_yield_curve"]= (df_copy.loc[i,'rconpc']-df_copy.loc[i-1,'rconpc'])/df_copy.loc[i-1,'rconpc']
        #df_copy.loc[i,"delta credit""]= (df_copy.loc[i,'rconpc']-df_copy.loc[i-1,'rconpc'])/df_copy.loc[i-1,'rconpc']
        


# ### Crisis-Dummy 

# Creation of a dummy variable indicating a crisis in t+1 or t+2
# 
# $$ 
# D = \left\{\begin{matrix} 1 \text{ if crisis in t+1 $\cap$ crisis in t+2} \\ 0 \text{ if no crisis in t+1 $\cap$ crisis in t+2} \end{matrix}\right.\
# $$
# 

# In[24]:


# temporary array of zeros, dimension number of rows in database
temp_array=np.zeros(len(df_copy))
# loop to create dummy
for i in np.arange(0,len(df_copy)-2):
    if (df_copy.loc[i+2,'iso'] != df_copy.loc[i,'iso']):
        temp_array[i]=np.nan
    else:
        temp_array[i]= 1 if ( (df_copy.loc[i+1,'crisisJST']== 1) or (df_copy.loc[i+2,'crisisJST']== 1)  ) else 0

#put the dummy in the dataframe

df_copy["crisis_warning"]=temp_array


# ### Keeping relevant data only 

# Reduction of Dataframe to only include the desired 12 pedictors (X) and y

# <font color='red'>if you have created "global_slope_yield_curve" and "global_delta_credit" as in remark 2, don't forget to include them</font>
# 
# <font color='red'>I included them -> 12 instead of 10</font>
# 
# -> Include them when upper part is fixed "global_slope_yield_curve", "global_delta_credit",

# In[25]:


variables=["slope_yield_curve","delta_credit","delta_debt_serv_ratio","delta_investm_ratio","delta_pdebt_ratio",           "delta_bmoney_gdp","delta_curr_acc_gdp","growth_cpi","growth_cons","eq_tr",            "crisis_warning"]

df_final=df_copy[variables].dropna()

# let's also create a version of our dataframe which includes the year
df_final_withyear=df_copy[["year"]+variables].dropna()


# # Functions necessary

# In[26]:


#function needed to plot ROC curve in exercise 3
def plot_roc(mod, X, y):
    # predicted_probs is an N x 2 array, where N is number of observations and 2 is number of classes
    predicted_probs = mod.predict_proba(X_test)

    # keep the second column, for label=1
    predicted_prob1 = predicted_probs[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, predicted_prob1)

    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")

    return fig, ax


# # Analysis

# In[27]:


print(df_copy.info())


# ## 1) Split sample (test and training sets)

# In[28]:


X = df_final.drop("crisis_warning", axis=1)
print("X.columns")
print(X.columns) #little redundant as X=df  

# First we will take a look at the dataset to get an impression of its structure first -> df.shape will return (x1, x2), where x1=nrrows and x2=nrcolumns
N=len(X)
print("number of observation = " + str(N))
print("shape")
print(df.shape)
print("head")
print(df.head())
print("df_final.columns")
print(df_final.columns)


# In[29]:


y = df_final["crisis_warning"] 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=percent_test)


# ## 2) Fitting the modules

# ### 2.1.) Logistic regression

# Remember from previous code that:
# - X = df_final.drop("crisis_warning", axis=1) (code line 14)
# - y = df_final["crisis_warning"] (code line 15)

# In[30]:


logistic_model = linear_model.LogisticRegression(solver="lbfgs")
logistic_model.fit(X_train, y_train)

beta_0 = logistic_model.intercept_[0]
beta_1 = logistic_model.coef_[0][0]
print(f"Logistic regression: y(x) = {beta_0:.4f} + {beta_1:.4f} X")


# Alternative version in which we can see all coefficients individually (better option for answering question 5):

# In[31]:


logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit()
print(result.summary2())

# used the instructions of the following website: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8


# ### 2.2.) Logistic regression with LASSO

# In[32]:


n_folds = 5
# candidate regularization parameters, smaller means heavier penalty, thus coefficients more shrinked to zero.
C_values = [0.001, 0.01, 0.05, 0.1, 1., 100.]
# define model
my_l1reg_logistic = LogisticRegressionCV(Cs=C_values, cv=n_folds, penalty='l1', 
                           refit=True, scoring='roc_auc', 
                           solver='liblinear', random_state=0,
                           fit_intercept=True)
# fit the model
my_l1reg_logistic.fit(X_train, y_train)
# these are already the best coefficients
coefs = my_l1reg_logistic.coef_
# mean of scores of class "1"
scores = my_l1reg_logistic.scores_[1]
mean_scores = np.mean(scores, axis=0)
# from this, you can visually inspect which C_value has the highest average score, thus is selected by the cross-validation
coefs


# ### 2.3.) Random trees

# ### 2.4.) Random Forests
# Using a loop, the model is fittet to a random forest including 1, 2, 3, 4, and 5 trees.
# For each random forest, the 
# - feature importances and the
# - confusion_matrix
# 
# are reported. 
# 
# #### Feature Importance 
# The feature importance - as the name indicates - shows us how important a respective feature (i.e. independent variable) is for the final prediction of the model. 
# The individual feature importance add up to 1 (i.e. 100%) and thus represent percentage shares. 
# 
# #### Confusion Matrix
# By definition a confusion matrix $C$ is such that $C_{i,j} $is equal to the number of observations known to be in group $i$ and predicted to be in group $j$.
# 
# Thus in binary classification, the count of 
# true negatives is $C_{0,0} $, false negatives is $C_{1,0} $, 
# true positives is  $C_{1,1} $and false positives is $C_{0,1}$. -- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">directly quoted from sklearn.org</a>.
# 
# Consequently, the values on the diagonal were estimated correctly. 
# The remaining values represent the type 1 and type 2 error respectively (in absolute terms). 

# In[33]:


for i in np.arange(1, 5): # number of trees in the forest
     print("number of trees is equal to "+ str(i))
     model4=RandomForestClassifier(n_estimators=i)# use Classifier instead of Regression, as this is a classification problem!
     model4.fit(X_train, y_train)
     print("model score for " + str(i) + " trees: " + str(model4.score(X_test, y_test)))
     print("Return the feature importances for " + str(i) + " trees: ")
     print(model4.feature_importances_)
     print("Return confusion_matrix for " + str(i) + " trees: ")
     print(metrics.confusion_matrix(y_test, model4.predict(X_test)))


# Confusion Matrices 
# 
# for 1 trees: -> 49 errors
#  
# for 2 trees: -> 23 errors
#  
#  for 3 trees: -> 31 errors
#  
#  for 4 trees: -> 18 errors 

# ### 2.5.) Neural Networks

# In class, we used the lbfgs-solver, which is an optimizer in the family of quasi-Newton methods.
# Here, we chose to use the state-of-the-art adam-solver instead. 
# The Adam optimization algorithm is an extension to stochastic gradient descent that has recently seen broader adoption for deep learning applications in computer vision and natural language. -- <a processing.href="https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/">the latter directly quoted from sklearn.org</a>.
# 
# Given the fact that we have sample with a length of 51 each we decided to experiment with 
# - 1 or 2 hidden layers
# - 100, 500 or 1000 neurons per layer 
# 
# The loops are constructed in a way to consider each possible combination of these three numbers neurons per layer using a cartesian product.

# In[ ]:


hiddenlayers = 2
nrneurons_1stlayer = np.array([100, 500, 1000]) 
nrneurons_2ndlayer = np.array([100, 500, 1000]) 
# The ith element represents the number of neurons in the ith hidden layer.

for i, j in itertools.product(nrneurons_1stlayer, nrneurons_2ndlayer): #cartesian product
     print("number of hiddenlayers is equal to "+ str(hiddenlayers))
     print("number of neurons per layer is equal to "+ str(i) + " and "+ str(j) + "respect.")
     model5=neural_network.MLPClassifier((i, j), activation="logistic", verbose=True, solver="adam", alpha=0.0)
     model5.fit(X_train, y_train)
     mse_model5 = metrics.mean_squared_error(y, model5.predict(X))
     print(str(mse_model5))
          

# Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by constraining the size of the weights.
# https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html


hiddenlayers = 1
nrneurons_1stlayer = np.array([100, 500, 1000]) 

for i in nrneurons_1stlayer: 
     print("number of hiddenlayers is equal to "+ str(hiddenlayers))
     print("number of neurons per layer is equal to "+ str(i))
     model5=neural_network.MLPClassifier((i), activation="logistic", verbose=True, solver="adam", alpha=0.0)
     model5.fit(X_train, y_train)
     mse_model5 = metrics.mean_squared_error(y, model5.predict(X))
     print(str(mse_model5))


# However, we receive a converge warning. 
# 
# As we know,  neural networks are extremely scale sensitive. 
# For this reason, we try to improve our model performance by standardizing it.
# We start with 
# - one layer
# -  100, 500 or 1000 neurons per layer 
# 
# #### Standardization

# In[ ]:


hiddenlayers = 1
nrneurons_1stlayer = np.array([100, 500, 1000]) 

for i in nrneurons_1stlayer: 
     print("number of hiddenlayers is equal to "+ str(hiddenlayers))
     print("number of neurons per layer is equal to "+ str(i))
     model5_scaled=pipeline.make_pipeline(
        preprocessing.StandardScaler(),  # this will do the input scaling
        neural_network.MLPClassifier((i), activation="logistic", verbose=True, solver="adam", alpha=0.0)
        )
     model5_scaled.fit(X_train, y_train)
     mse_model5_scaled = metrics.mean_squared_error(y, model5_scaled.predict(X))
     print(str(mse_model5_scaled))


# A comparison of the scaled models MSE vs. the unscaled models MSE: 

# In[ ]:


print(f"Unscaled MSE {mse_model5}")
print(f"Scaled MSE {mse_model5_scaled}")


# Now we use the accuracy score instead of the MSE for further improvment. 
# It is a multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. -- <a processing.href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score">the latter directly quoted from sklearn.org</a>.

# In[ ]:


for i in nrneurons_1stlayer: 
     print("number of hiddenlayers is equal to "+ str(hiddenlayers))
     print("number of neurons per layer is equal to "+ str(i))
     model5_scaled=pipeline.make_pipeline(
        preprocessing.StandardScaler(),  # this will do the input scaling
        neural_network.MLPClassifier((i), activation="logistic", verbose=True, solver="adam", alpha=0.0)
        )
     model5_scaled.fit(X_train, y_train)
     accuracy_model5_scaled = metrics.accuracy_score(y, model5_scaled.predict(X))
     print(str(accuracy_model5_scaled))


# <ins>Summing up these results: </ins> <br />
# number of hiddenlayers is equal to 1 <br />
# number of neurons per layer is equal to 100 <br />
# -> Accuracy of 92,97297297297298%
# 
# 
# number of hiddenlayers is equal to 1 <br />
# number of neurons per layer is equal to 500 <br />
# -> Accuracy of 93,03303303303303%
# 
# number of hiddenlayers is equal to 1 <br />
# number of neurons per layer is equal to 1000 <br />
# -> Accuracy of 92,97297297297298%
# 
# 
# <ins>Take a moment to appreciate what you are seeing: </ins> <br />
# __Our neural network with 1 hidden layer and 500 neurons is able to predict a crisis with >93% accuracy and is thus our best prediction technique!__

# ### Taking stock

# <font color='red'>here, I would give a brief summary of our results so far </font>
# 

# ## 3) ROC curves

# In[ ]:


#Plot the ROC curves for the best versions of your models and compute the AUROC. 

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

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('True Value')


# ## 4) Comparison of the different models confusion matrices 
# By definition a confusion matrix $C$ is such that $C_{i,j} $is equal to the number of observations known to be in group $i$ and predicted to be in group $j$.
# 
# Thus in binary classification, the count of 
# true negatives is $C_{0,0} $, false negatives is $C_{1,0} $, 
# true positives is  $C_{1,1} $and false positives is $C_{0,1}$. -- <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html">directly quoted from sklearn.org</a>.
# 
# Consequently, the values on the diagonal were estimated correctly. 
# The remaining values represent the type 1 and type 2 error respectively (in absolute terms). 

# ### Confusion matrices of Random Forests

# In[ ]:


for i in np.arange(1, 5): # number of trees in the forest
     print("number of trees is equal to "+ str(i))
     model4=RandomForestClassifier(n_estimators=i)# use Classifier instead of Regression, as this is a classification problem!
     model4.fit(X_train, y_train)
     print("model score for " + str(i) + " trees: " + str(model4.score(X_test, y_test)))
     print("Return the feature importances for " + str(i) + " trees: ")
     print(model4.feature_importances_)
     print("Return confusion_matrix for " + str(i) + " trees: ")
     print(metrics.confusion_matrix(y_test, model4.predict(X_test)))


# Confusion Matrices 
# 
# for 1 trees: -> 49 errors
#  
# for 2 trees: -> 23 errors
#  
#  for 3 trees: -> 31 errors
#  
#  for 4 trees: -> 18 errors 

# In[ ]:


#-------- Confusion Matrices Comparison
#Plot the true value of y against the prediced value 
y_predicted = model.predict(X_test)
cm = confusion_matrix(y_test, y_predicted)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('True Value')

#According to this criterion, which model performs best?


# ## 5) Which variables 'survive' in the logistic regression with L1 regularization ?

# Considering the outputs in which we coded the logistic regression with L1 regularization, most coefficients remain at a value very different to 0. 
# Only the last coefficient (eq_tr), which refers to the Nominal Total Equity return. This means that this variable is not a good predictor for whether tere will be an economic crisis, whilst all others are and "survive" the L1 regularization.
# Comparing the coefficient values with and without the L1 regularization, we can see that the beta of each of the variables that survived has changed,including in some cases their sign.
# - The slope_of_the_yield_curve's coefficient went from a -39,67 to a -23,56. Meaning that even with the regularization, this variable still is quite significant for predicting crisis.
# - The delta_credit's coefficient went from a -5,42 to a 9,83. With the regularization, whilst the variable remains significant, its impact completly shifts. Whilst previously 1 year-variation of credit led to a lower likelihood of a crisis happening, the opposite occurs with the regularization.
# - The delta_debt_serv_ratio's coefficient went from a 35,46 to a 67,74. This means that with the L1 regularization the impact of the 1 year-variation of debt ser ratio on the likelihood of a crisis hapenning increased.
# - The delta_investm_ratio's coefficient went from a 15,97 to a 16,79.  After the L1 regularization, the impact of a 1 year-variation of investment/gdp ratio hasn't changed too much from what was initially predicted.
# - The delta_pdebt_ratio's coefficient went from a -3,72 to a -6,64. With the L1 regularization, the impact the 1 year-variation of public debt/gdp on the likelihood of a crisis happening has increased.
# - The delta_bmoney_gdp's coefficient went from a -12,11 to a 3,06. With the L1 regularization, the impact of a 1 year-variation of broad money / gdp has completly shifted. Taking the regularization into account, a higher variation of this variable leads to a higher probability of a crisis happening. We have the opposite effect if we don't consider this.
# - The delta_curr_acc_gdp's coefficient went from a -9,84 to a 1,30. After the L1 regularization, the impact of 1 year-variation of current / gdp on the likelihood of a crisis happening has shifted - it used to be negative and now is positive.
# - The growth_cpi's coefficient went from a -22,99 to a -10,65. The impact of annual inflation, whilst higher in magnitude, hasn't changed in terms of signal.

# In[ ]:




