#DELETE: Add info in title in the end!
#***********************************************************************************************
# Title:                 XXXXXX 
# Author:                Ana Margarida Silva da Costa & Lea Katharina Paoli
# Matriculation Nr.      XXXXXX & XXXXX
# Purpose:               XXXXXX 
#************************************************************************************************

##ask the user for path: 

#***********************IMPORT MODULES***********************

# Installation of packages [Use the command line, not the Python shell]
"""
py -m pip install numpy
py -m pip install --upgrade pandas-datareader
py -m pip install -U scikit-learn
pip install openpyxl
pip install seaborn
pip install plotly

#check whether the installation worked properly: 
py -m pip list
for help, see: https://phoenixnap.com/kb/install-pip-windows 
& https://www.youtube.com/watch?v=SrX5yo4KKGM 
& https://www.youtube.com/watch?v=RvbUqf3Tb1s
"""

# Import the Python modules you need
import pickle # allows us to store training date into a file 

import numpy as np
import pandas as pd 
import plotly.graph_objects as go

from sklearn import (
    model_selection, linear_model, ensemble, metrics, neural_network, pipeline, model_selection, \
    tree, preprocessing, pipeline
)
# the pipeline defines any number of steps that will be applied to transform the `X` data 
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
percent_test = 0.25
print("percent_test is set to " + str(percent_test))

# Input Data UPDATE LOCATION OF FILE AS NEEDED 
path_data = r"C:\Users\leapa\PycharmProjects\AdvFinEcon_PythonAssignment\JSTdatasetR4.xlsx"

print("Your excel file should be stored here: " + path_data)

# Cross-Validation Splits
number_of_splits = 5
kf = StratifiedKFold(n_splits=number_of_splits)
print("number_of_splits is set to " + str(number_of_splits))

#***********************FUNCTIONS***********************
#DELETE PLEASE LOOK HERE:
# these functions are needed to plot the trees-graphs 
def Ey_x(x):
    return 1/3*(np.sin(5*x[0])*np.sqrt(x[1])*np.exp(-(x[1]-0.5)**2)) #DELETE: PLEASE LOOK HERE: WTF is this for?

def surface_scatter_plot(X,y,f, xlo=0., xhi=1., ngrid=50, width=860, height=700, f0=Ey_x, show_f0=False): #PLEASE LOOK HERE: f0=Ey_x?!
    scatter = go.Scatter3d(x=X[:,0],y=X[:,1],z=y,
                           mode='markers',
                           marker=dict(size=2, opacity=0.3)
                        )

    xgrid = np.linspace(xlo,xhi,ngrid)
    ey = np.zeros((len(xgrid),len(xgrid)))
    ey0 = np.zeros((len(xgrid),len(xgrid)))
    colorscale = [[0, colors[0]], [1, colors[2]]]
    for i in range(len(xgrid)):
        for j in range(len(xgrid)):
            ey[j,i] = f([xgrid[i],xgrid[j]])
            ey0[j,i]= f0([xgrid[i],xgrid[j]])
    
    surface = go.Surface(x=xgrid, y=xgrid, z=ey, colorscale=colorscale, opacity=1.0)
    if (show_f0):
        surface0 = go.Surface(x=xgrid, y=xgrid, z=ey0, opacity=0.8, colorscale=colorscale)
        layers = [scatter, surface, surface0]
    else:
        layers = [scatter, surface]
    
    fig = go.FigureWidget(
        data=layers,
        layout = go.Layout(
            autosize=True,
            scene=dict(
                xaxis_title='X1',
                yaxis_title='X2',
                zaxis_title='Y'
            ),
            width=width,
            height=height,
            template=plotly_template,
        )
    )
    return fig


#***********************PREPATORY WORK***********************
print("***********************PREPATORY WORK***********************")
#Load the dataset as a Pandas Data Frame
df=pd.read_excel(path_data, sheet_name="Data", engine='openpyxl') #engine had to be added to open xlxs file (Source: https://stackoverflow.com/questions/65250207/pandas-cannot-open-an-excel-xlsx-file)


#DELETE: ATTENTION THIS PART HAD TO BE CHANGED!!!!!
#let's make a copy, in order to preserve original dataset
df_copy=df.copy()
#let's create new (temporary) columns with the transformed variables we need:
#-slope of the yield curve
df_copy["slope_yield_curve"]=df_copy["ltrate"]/100-df_copy["stir"]/100
# credit: loans to the private sector / gdp
df_copy["credit"]=df_copy["tloans"]/df_copy["gdp"]
# debt service ratio: credit * long term interest rate
df_copy["debt_serv_ratio"]=(df_copy["tloans"]/df_copy["gdp"])*df_copy["ltrate"]/100
# broad money over gdp
df_copy["bmoney_gdp"]=df_copy["money"]/df_copy["gdp"]
# current account over gdp
df_copy["curr_acc_gdp"]=df_copy["ca"]/df_copy["gdp"]
# Now we need to compute 1-year absolute variations and percentage variations for a few variables
# create 1 year-variation of credit from grouped dataframe and add back to initial dataframenp.zeros
# create 1 year-variation of debt ser ratio from grouped dataframe and add back to initial dataframe
# create 1 year-variation of investment/gdp from grouped dataframe and add back to initial dataframe
# create 1 year-variation of public debt/gdp from grouped dataframe and add back to initial dataframe
# create 1 year-variation of broad money / gdp from grouped dataframe and add back to initial dataframe
# create 1 year-variation of current / gdp from grouped dataframe and add back to initial dataframe

df_copy["delta_credit"]=np.empty(len(df_copy))*np.nan
df_copy["delta_debt_serv_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_investm_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_pdebt_ratio"]=np.empty(len(df_copy))*np.nan
df_copy["delta_bmoney_gdp"]=np.empty(len(df_copy))*np.nan
df_copy["delta_curr_acc_gdp"]=np.empty(len(df_copy))*np.nan
df_copy["growth_cpi"]=np.empty(len(df_copy))*np.nan
df_copy["growth_cons"]=np.empty(len(df_copy))*np.nan


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



# NOW JUMP TO REMARK 2 AT THE BOTTOM IF YOU WANT TO INCLUDE GLOBAL CREDIT AND SLOPE OF THE YIELD CURVE AS PREDICTORS
#DELETE: TO BE CHECK! This was added by me from remark to, is that correct?


# first create a temporary column including the sum of GDP for each country grouped by year of observation
# check if the column name is actually called year
df_copy["sum_gdp_by_year"]=df_copy.groupby("year")["gdp"].transform('sum')
# now create a new temporary column with the slope of the yield curve for each observation multiplied
# by the gdp weigh
df_copy["weighted_slope"]=df_copy["slope_yield_curve"]*df_copy["gdp"]/df_copy["sum_gdp_by_year"]
# finally create the column "global_slope_yield_curve" by summing the weighted domestic slopes,
# after grouping observation by year
df_copy["global_slope_yield_curve"]=df_copy.groupby("year")["weighted_slope"].transform('sum')


#DELETE: TO BE CHECK! This was added by me from remark to, is that correct? END

# low let's create the crises early warning label: a dummy variable which takes value one if in the next 
# year or the next two years there will be a crises

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

# create a smaller dataframe including only the variables we are interested in: the first ten are predictors (X) and the last one is the output, or label (y)
# if you have created "global_slope_yield_curve" and "global_delta_credit" as in remark 2, don't forget to include them
variables=["slope_yield_curve","delta_credit","delta_debt_serv_ratio","delta_investm_ratio","delta_pdebt_ratio","delta_bmoney_gdp","delta_curr_acc_gdp","growth_cpi","growth_cons","eq_tr","crisis_warning"]
df_final=df_copy[variables].dropna()

# let's also create a version of our dataframe which includes the year
df_final_withyear=df_copy[["year"]+variables].dropna()

#DELETE: ATTENTION THE PART ABOVE HAD TO BE CHANGED!!!!!

#***********************ANALYSIS***********************

print(df.info())
#-------- Split Sample
# Training vs. Test (randomly assigned)
# look at dataset to get an impression of its structure first -> df.shape will return (x1, x2), where x1=nrrows and x2=nrcolumns
print("shape")
df.shape
print("head")
df.head()
print("df_final.columns")
print(df_final.columns)
X = df_final.drop("crisis_warning", axis=1)
print("X.columns")
print(X.columns) #little redundant as X=df 

y = df_final["crisis_warning"]
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=percent_test)

#DELETE: https://www.youtube.com/watch?v=fwY9Qv96DJY
#train_test_split(df[['X {= something multidimensional -> 2 brackets}']], df.Y{=explanatory variable}, test_size={percentage of the sample used for testing, e.g. 0,1})
# add "random_state=" to keep sample allocation fixed when the code is executed multiple times
#results we get back from train_test_split(): X_train, X_test, y_train, y_test

 
# Varify
N=len(X)
print("number of observation = " + str(N))
N2= len(X_test) + len(X_train)
print(N2)
ptimesN=percent_test*N
oneminusptimesN=(1-percent_test)*N
print("number of obs. in test data set should be " + str(ptimesN))
print("number of obs. in test data set is " + str(len(X_test))) #should equal to (1-percent_test)*N
print("number of obs. in training data set should be " + str(oneminusptimesN))
print("number of obs. in training data set is " + str(len(X_train))) #should equal to percent_test*N


#-------- Fit Model

#PLEASE LOOK HERE:
#ANA MARGARIDAS Part 
# We will ne this weird function "surface_scatter_plot" it's defined above in the section "functions"
# It comes from the regression-notebook the prof shared
# HOWEVER: I cannot run it - neither there nor on my local python. The TA managed to do it if I remember correctly 


fitted_tree = tree.DecisionTreeClassifier(max_depth=3).fit(Xsim,ysim)
fig=surface_scatter_plot(Xsim, ysim, lambda x: fitted_tree.predict([x]), show_f0=True) #lamdba refers to the "anonymous" lambda funtion
fig.show() 

#DELETE: LOOK HERE: THIS IS REMARK 1 -> I think this needs to be integrated into your part somewhere 
# The terminology 'logistic regression with LASSO regularization is a bit misleading. I should have said 'l1' regularization, 
# used to penalize the absolute value of coefficients thus shrinking them towards zero. In order to implement this with Sklearn, 
# you can do the following:

# n_folds = 5
# # candidate regularization parameters, smaller means heavier penalty, thus coefficients more shrinked to zero.
# C_values = [0.001, 0.01, 0.05, 0.1, 1., 100.]
# # define model
# my_l1reg_logistic = LogisticRegressionCV(Cs=C_values, cv=n_folds, penalty='l1', 
#                            refit=True, scoring='roc_auc', 
#                            solver='liblinear', random_state=0,
#                            fit_intercept=True)
# # fit the model
# my_l1reg_logistic.fit(X_train, y_train)
# # these are already the best coefficients
# coefs = my_l1reg_logistic.coef_
# # mean of scores of class "1"
# scores = my_l1reg_logistic.scores_[1]
# mean_scores = np.mean(scores, axis=0)
# # from this, you can visually inspect which C_value has the highest average score,
# # thus is selected by the cross-validation


# #--- MODEL4: random forest
# # DELETE: https://www.youtube.com/watch?v=ok2s1vV9XW0
# for i in np.arange(1, 5): # number of trees in the forest
#     print("number of trees is equal to "+ str(i))
#     model4=RandomForestClassifier(n_estimators=i)# use Classifier instead of Regression, as this is a classification problem!
#     #DELETE: https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/
#     model4.fit(X_train, y_train)
#     print("model score for " + str(i) + "trees: " + str(model4.score(X_test, y_test)))
#     model4.feature_importances_


#     fig=surface_scatter_plot(X_train,y_train,lambda x: forest.predict([x]), show_f0=True)
#     fig.show()

# drawing single decision graph no longer possible 
# Feature importance is the average MSE decrease caused by splits on each feature.
# If a given feature has greater importance, the trees split on that feature more often and/or splitting on that feature resulted in larger MSE decreases.



## MODEL5: neural networks.
# Experiment with different numbers of hidden layers, and neurons for each layers, not necessarily using cross-validation
from sklearn import neural_network
model5 = neural_network.MLPClassifier((6,), activation="logistic", verbose=True, solver="lbfgs", alpha=0.0)
#‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
# Alpha is a parameter for regularization term, aka penalty term, that combats overfitting by constraining the size of the weights.
# https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
model5.fit(X_test,y_test)

fig=surface_scatter_plot(Xsim,ysim,lambda x: nn.predict([x]), show_f0=True)
fig.show()

# two hidden layers, with N1=30 and N2=20
nn_model = neural_network.MLPClassifier((30, 20))
nn_model.fit(X_test, y_test)
#https://stackoverflow.com/questions/62658215/convergencewarning-lbfgs-failed-to-converge-status-1-stop-total-no-of-iter

#ax = var_scatter(df)
#scatter_model(nn_model, X, ax=ax)

mse_nn = metrics.mean_squared_error(y, model5.predict(X))


# Standardize 
nn_scaled_model = pipeline.make_pipeline(
    preprocessing.StandardScaler(),  # this will do the input scaling
    neural_network.MLPClassifier((30, 20)) 
)

# We can now use `model` like we have used our other models all along
# Call fit
nn_scaled_model.fit(X_test, y_test)

# Call predict
mse_nn_scaled = metrics.mean_squared_error(y, nn_scaled_model.predict(X))
#mse_nn / metrics.mean_squared_error(y, lr_model.predict(X)) NOT YET POSSIBLE AS LINEAR MODEL IS MISSING

print(f"Unscaled mse {mse_nn}")
print(f"Scaled mse {mse_nn_scaled}")

#ax = var_scatter(df)
#scatter_model(nn_scaled_model, X, ax=ax)

#-------- ROC curves
#Plot the ROC curves for the best versions of your models and compute the AUROC. 
#DELETE: https://www.youtube.com/watch?v=gJo0uNL-5Qw
#See classification file

def plot_roc(mod, X, y):
    # predicted_probs is an N x 2 array, where N is number of observations
    # and 2 is number of classes
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

plot_roc(logistic_age_model, X_test, y_test)


#AUC Value (AUROC?!)
predicted_prob1 = logistic_age_model.predict_proba(X)[:, 1]
auc = metrics.roc_auc_score(y, predicted_prob1)
print(f"Initial AUC value is {auc:.4f}")


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