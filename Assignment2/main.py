# Installation of packages
pip install numpy
pip install --upgrade pandas-datareader

# Import the Python modules you need
import numpy as np
import pandas as pd


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

#Analysis