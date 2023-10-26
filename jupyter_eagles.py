# -*- coding: utf-8 -*-


# Libraries Install
from fitter import Fitter
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import scipy.stats as stats
import requests
from ydata_profiling import ProfileReport

# df_cleaned = combined_df.dropna(axis=0, thresh=20)
# df_cleaned['Offense']=df_cleaned['Offense'].fillna(0)
# df_cleaned['Defense']=df_cleaned['Defense'].fillna(0)
# df_cleaned['Win']=np.where(df_cleaned['Unnamed: 5_level_0']=='W',1,0)
# df_cleaned['Unnamed: 8_level_0']=df_cleaned['Unnamed: 8_level_0'].fillna(0)
# df_cleaned['Home_Games']=np.where(df_cleaned['Unnamed: 8_level_0']=='@',1,0)
# df_cleaned=df_cleaned.drop(columns=['Unnamed: 8_level_0', 'Unnamed: 6_level_0'], axis=1)

# df_cleaned.to_csv('Eagles_Data_Final.csv')
# Some header cleaning such that ther are no subheaders

# eagles=pd.read_csv("/Users/omerabdelrahim/Downloads/Eagles_Data_Final.csv", sep=";")
# eagles['MOV']=eagles['TmScore']-eagles['OppScore']
# eagles['MOT']=eagles['DTO']-eagles['OTO']
# eagles=eagles.drop(['Rec'], axis=1)
# eagles.to_csv('Eagles_10yr.csv')

"""#1. Descriptive Analysis: Perform a univariate analysis following the steps below.

We will be analyzing data for the Philadelphia Eagles Football Team from the years 2013 to 2022. Our goal is to ultimately find out the most appropriate variables that contributed to Margin Of Victory (MOV) for the team.

## (a) Begin by providing a descriptive analysis of your variables (include all predictors and response variable). This should include things like histograms, quantile plots, correlation plots, etc.

### Variable Names:

MOV= Margin of Victory

MOT = Margin of Turnover

O1stD = First Downs Gained

D1stD = First Downs Allowed

DRushY = Rush Yards Allowed

ORushY = Rush Yards Gained

Home_Games = Indicator Variable for whether the game was played at home

EOffense = Expected Offensive Points

EDefense = Expected Defensive Points

ESpTms = Expected Specials Teams Points
"""

eagles=pd.read_csv('Eagles_10yr (1).csv')
eagles.describe()

# Replace the commas in the data with periods
eagles['EOffense'] = eagles.EOffense.str.replace(',', '.')
eagles['EDefense'] = eagles.EDefense.str.replace(',', '.')
eagles['ESpTms'] = eagles.ESpTms.str.replace(',', '.')

# Convert the objects to floats
eagles['EOffense'] = eagles.EOffense.astype(float)
eagles['EDefense'] = eagles.EDefense.astype(float)
eagles['ESpTms'] = eagles.ESpTms.astype(float)

print(eagles.columns)

print(type(eagles))

# Check for null values
eagles.info()

# Count the number of null values
eagles.isnull().sum()

# Initial linear model test
model1 = smf.ols(formula='MOV~MOT+DRushY+ORushY+O1stD+D1stD+Home_Games', data=eagles)
model1_results=model1.fit(cov_type='HC1')
print(model1_results.summary())

# Histograms and Density plots

plt.figure(figsize = (10,6))
sns.histplot(eagles.MOV, stat = "density")
sns.kdeplot(eagles.MOV, color = "red")
plt.title("Philadelphia Eagles Margin of Victory 2013-2022")
plt.xlabel("Margin of Victory")
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.MOT, stat = "density")
sns.kdeplot(eagles.MOT, color = "red")
plt.title("Philadelphia Eagles Margin of Turnovers 2013-2022")
plt.xlabel("Margin of Turnovers")
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.O1stD, stat = "density")
sns.kdeplot(eagles.O1stD, color = "red")
plt.title("Philadelphia Eagles 1st Downs Gained 2013-2022")
plt.xlabel("1st Downs Gained")
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.D1stD, stat = "density")
sns.kdeplot(eagles.D1stD, color = "red")
plt.title("Philadelphia Eagles 1st Downs Allowed 2013-2022")
plt.xlabel("1st Downs Allowed")
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.DRushY, stat = "density")
sns.kdeplot(eagles.DRushY, color = "red")
plt.title("Philadelphia Eagles Rush Yards Allowed 2013-2022")
plt.xlabel("Rush Yards Allowed")
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.ORushY, stat = "density")
sns.kdeplot(eagles.ORushY, color = "red")
plt.title("Philadelphia Eagles Rush Yards Gained 2013-2022")
plt.xlabel("Rush Yards Gained")
plt.show()

"""The density curve on the above plots appears to show a normal distribution. The variables DRushY and ORushY appear to show more skewness compared to the rest of the variables. These might need to be transformed."""

# QQ-Plots

stats.probplot(eagles.MOV, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Victory 2013-2022")
plt.show()

stats.probplot(eagles.MOT, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Turnover 2013-2022")
plt.show()

stats.probplot(eagles.O1stD, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Turnover 2013-2022")
plt.show()

stats.probplot(eagles.D1stD, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Turnover 2013-2022")
plt.show()

stats.probplot(eagles.DRushY, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Turnover 2013-2022")
plt.show()

stats.probplot(eagles.ORushY, dist="norm", plot=plt)
plt.title("Philadelphia Eagles Margin of Turnover 2013-2022")
plt.show()

"""The data in the six variables appears to be normal for the most part even though some deviations from the x=y line can be observed. More quantitative analysis is needed to conclude normality."""

# Correlation Plot
r_vars=eagles[['MOT','MOV','O1stD','D1stD','ORushY','DRushY','Home_Games']]

plt.figure(figsize=(13,7))
data=r_vars
c= data.corr()
sns.heatmap(c,cmap="Blues",annot=True,square = True)

"""As can be viewed in this correlation plot "heatmap" some of the strongest relative correlations are between MOV and MOT as well as between O1stD and ORushY and vice a versa for the defensive statistics"""

# Pair Plot
sns.pairplot(r_vars, kind='reg')

"""## (b) Discuss your findings from doing an exploratory analysis using Pandas Profiling. Did you discover anything new?"""

ProfileReport(r_vars)

"""The report shows results similar to what we found in 1(a) e.g., number of variables, number of observations and missing cells etc. One new discovery is that the report states that the variable MOT is highly overall correlated with MOV and vice versa. We noticed from our correlation plot that the value is 0.55

## (c) Estimate density distributions (e.g., Cullen & Frey) for all your variables, and show the plots with the respective fits.
"""

# We will be using MOV, MOT, EOffense, EDefense, ESpTms, DRushY,ORushY, O1stD, D1stD
d_MOV = sns.kdeplot(eagles.MOV)
plt.legend(['d_MOV'])
plt.title('Density Distributions of MOV')
plt.show()

d_MOT = sns.kdeplot(eagles.MOT)
plt.legend(['d_MOT'])
plt.title('Density Distributions of MOT')
plt.show()

d_EOffense = sns.kdeplot(eagles.EOffense)
plt.legend(['d_EOffense '])
plt.title('Density Distributions of EOffense')
plt.show()

d_EDefense = sns.kdeplot(eagles.EDefense)
plt.legend(['d_EDefense'])
plt.title('Density Distributions of EDefense')
plt.show()

d_ESpTms = sns.kdeplot(eagles.ESpTms)
plt.legend(['d_ESpTms'])
plt.title('Density Distributions of ESpTms')
plt.show()

d_DRushY = sns.kdeplot(eagles.DRushY)
plt.legend(['d_DRushY'])
plt.title('Density Distributions of DRushY')
plt.show()

d_ORushY = sns.kdeplot(eagles.ORushY)
plt.legend(['d_ORushY'])
plt.title('Density Distributions of ORushY')
plt.show()

d_O1stD = sns.kdeplot(eagles.O1stD)
plt.legend(['d_O1stD'])
plt.title('Density Distributions of O1stD')
plt.show()

d_D1stD = sns.kdeplot(eagles.D1stD)
plt.legend(['d_D1stD'])
plt.title('Density Distributions of D1stD')
plt.show()

fig = plt.figure(figsize = (12,5))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)

ax1.hist(eagles.MOT, color = 'r')
ax2.hist(eagles.MOV, color = 'g')
ax3.hist(eagles.DRushY, color = 'b')
ax4.hist(eagles.ORushY, color = 'y')
ax5.hist(eagles.O1stD, color = 'purple')
ax6.hist(eagles.D1stD, color = 'grey')

# Fitter Plots for fitting distributions onto variables

f = Fitter(eagles.MOV)
f.fit()
f.summary()

f = Fitter(eagles.MOT)
f.fit()
f.summary()

f = Fitter(eagles.EOffense)
f.fit()
f.summary()

f = Fitter(eagles.EDefense)
f.fit()
f.summary()

f = Fitter(eagles.ESpTms)
f.fit()
f.summary()

f = Fitter(eagles.DRushY)
f.fit()
f.summary()

f = Fitter(eagles.ORushY)
f.fit()
f.summary()

f = Fitter(eagles.O1stD)
f.fit()
f.summary()

f = Fitter(eagles.D1stD)
f.fit()
f.summary()

# We will be using MOV, MOT, EOffense, EDefense, ESpTms, DRushY,ORushY, O1stD, D1stD
d_MOT = sns.kdeplot(eagles.MOT)
plt.legend(['d_MOT'])
plt.title('Density Distributions of Chose Variables')
plt.show()

plt.figure(figsize = (10,6))
sns.histplot(eagles.ORushY, stat = "density")
sns.kdeplot(eagles.ORushY, color = "red")

pd.set_option('display.max_columns', 26)
pd.set_option('display.max_rows', 8)
eagles.describe()

"""##(d) Identify if there are any non-linearities within your variables. What transformations should you perform to make them linear? What would happen if you included nonlinear variables in your regression models without transforming them first?

Based on the graphical results from the pair plots as well as panda_profiling in section (b) the data appears to be linear. We will need to perform a transformation of the two variables DRushY and ORushY. The results of the transformation using Box-cox will be shown in the later section to follow.
"""

# Box-Cox Transformation of DRushY and ORushY
import scipy


bc_DRY,lambda_DRY = scipy.stats.boxcox(eagles["DRushY"])
print(lambda_DRY)

sns.histplot(eagles["DRushY"])
plt.title("Original DRushY")
plt.show()

sns.histplot(bc_DRY)
plt.title("Box-Cox Transformed: DRushY")
plt.show()

bc_ORY,lambda_ORY = scipy.stats.boxcox(eagles["ORushY"])
print(lambda_ORY)

sns.histplot(eagles["ORushY"])
plt.title("Original ORushY")
plt.show()

sns.histplot(bc_ORY)
plt.title("Box-Cox Transformed: ORushY")
plt.show()

"""Upon comparison of the two variables before and after the transformation - the results show that the distributions appears normal."""

# Here, we append the original dataset with the tranformed variables
eagles['BCORY']=bc_ORY
eagles['BCDRY']=bc_DRY
# New dataset r_vars
r_vars=eagles[['MOT','MOV','O1stD','D1stD','BCORY','BCDRY','Home_Games']]

# Graphical Represenation of the transformed variables
sns.lmplot(data=r_vars, x='BCORY', y='MOV', line_kws={'color':'red'}, lowess=False, height=5, aspect=1)
plt.grid()
sns.lmplot(data=r_vars, x='BCDRY', y='MOV', line_kws={'color':'green'}, lowess=False, height=5, aspect=1)
plt.grid()

# Pair Plot
sns.pairplot(r_vars, kind='reg')

"""The new pair plot with the transformed variables BCDRY and BCORY shows these two variables with a normal distribution.

If non-linear variables are included in the model the results will not be robust and reliable in comparison to those of a model with linear variables.

##(e) Comment on any outliers and/or unusual features of your variables.

In the graphs below we will see very few outliers (less than 5%) but we do not find it appropriate to remove these outliers due to the nature of data. We think the removal of outliers will distort the findings of our analysis.
"""

# Outliers
sns.lmplot(data = eagles, x = 'MOT', y = 'MOV')
plt.title("MOT Regressed")


sns.lmplot(data = eagles, x = 'DRushY', y = 'MOV')
plt.title("DRushY Regressed")


sns.lmplot(data = eagles, x = 'ORushY', y = 'MOV')
plt.title("ORushY Regressed")

sns.lmplot(data = eagles, x = 'O1stD', y = 'MOV')
plt.title("O1stD Regressed")


sns.lmplot(data = eagles, x = 'D1stD', y = 'MOV')
plt.title("D1st Regressed")

plt.show()

"""## (f) If you have any NAs, impute them using any of the methods discussed in class but make sure to justify your choice.

We do not have any NAs in our dataset as was confirmed in part 1(a)
"""

# Boxplots to show outliers
plt.boxplot(eagles.MOV)
plt.grid

"""The above box plot clearly indicates the presence of outliers.

# 2. Variable Selection:

## (a) Using the Boruta Algorithm identify the top 2 predictors
"""

from BorutaShap import BorutaShap

from sklearn.ensemble import RandomForestRegressor

boruta_data = eagles[[ 'MOV','O1stD','BCORY','D1stD','BCDRY','MOT', 'Home_Games']].copy()

x = boruta_data.iloc[:,1:]

x = boruta_data.iloc[:, 1:]
y = boruta_data['MOV']
Feature_Selector = BorutaShap(importance_measure='shap', classification=False)
Feature_Selector.fit(X=x, y=y, n_trials=100, random_state=0)
Feature_Selector.plot(which_features='all')

Feature_Selector.Subset()

from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
x = boruta_data.iloc[:, 1:]
y = boruta_data['MOV']
xCols = x.columns.tolist()

currentTrainX = x.to_numpy()
currentTrainY = y.to_numpy().ravel()

forest = RandomForestRegressor(n_jobs=-1, max_depth = 5)
forest.fit(currentTrainX, currentTrainY)

np.int = int
np.float = float
np.bool = bool
boruta = BorutaPy(forest, n_estimators='auto', verbose=2, random_state=1)
boruta.fit(currentTrainX, currentTrainY)

featureSupport = list((zip(xCols, boruta.support_)))
featureSupport

# Ranking Boruta results
boruta.ranking_
# for better visualization use of the boruta ranking
featureRanks = list(zip(xCols, boruta.ranking_))
sorted(featureRanks, key=lambda x: x[1])

"""The top ranking variables from our Boruta Algorithm are the Box-Cox Transformation of Offensive (BCORY) and Defensive (BCDRY) Rushing Yards as well as MOT and Defensive 1st Downs Allowed (D1stD).
The top 2 are BCORY and BCDRY

Going forward it would be prudent to work with one of these 4 explanatory variables, and we will use Mallows CP in order to come to a clearer solution.

## (b) Using Mallows Cp identify the top 2 predictors
"""
# Mallows CP loops and Regression construction

from RegscorePy import mallow

model = smf.ols(formula='(MOV) ~ MOT + O1stD + BCORY + D1stD + BCDRY', data=eagles)
results = model.fit(cov_type='HC1')
results.summary()

import itertools

model = smf.ols(formula='MOV ~ MOT + O1stD + BCORY + D1stD + BCDRY', data=eagles)
results = model.fit()
y = eagles['MOV']
y_pred=results.fittedvalues


storage_cp = pd.DataFrame(columns = ["Variables", "CP"])
k = 8

for L in range(1, len(r_vars.columns[0:]) + 1):
    for subset in itertools.combinations(r_vars.columns[0:], L):

        formula1 = 'MOV~'+'+'.join(subset)

        results = smf.ols(formula=formula1, data = eagles).fit()
        y_sub = results.fittedvalues
        p = len(subset)+1

        cp = mallow.mallow(y, y_pred,y_sub, k, p)

        storage_cp = storage_cp._append({'Variables': subset, 'CP': cp}, ignore_index = True)

pd.set_option('display.max_columns', 2)
pd.set_option('display.max_rows', 127)
storage_cp.sort_values(by = "CP")

"""Based on the Mallows CP report we find that our best predictors of Margin of Victory (MOV) are Margin of Turnover (MOT) and Defensive 1st Downs Allowed (D1stD). We still observe very large Mallows CP values with these single explanatory variables relative to multivariable regression models.

# 3. Model Building: Explore several competing OLS models (based on part 2) and decide on one model only (with just one predictor). You will need to explain in detail how you arrived at your preferred model. Discuss the economic significance of your parameters, and overall findings. Make sure you discuss your main conclusions and recommendations. At a minimum. you need to include the following checks:
"""

# Specify the Model usings results from Mallow cp
ols_mod = smf.ols(formula='MOV ~ MOT', data = eagles)

# Fit the Model
ols_fit = ols_mod.fit()

# Look at the Model Fit Summary
print(ols_fit.summary())

pip install simple_colors

# Evaluating the model
import statsmodels.stats.api as sms
from simple_colors import *

# Linearity: Harvey-Collier --> Ho: model is linear
name = ["t-stat", "p-value"]
test = sms.linear_harvey_collier(ols_fit)
print(blue("Linearity Test Results:",['bold']))
print(list(zip(name, test)))
print("\n")

# Normaility of the Residuals: Jarque-Bera --> Residuals ~ N(0,1)
name = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
test = sms.jarque_bera(ols_fit.resid)
print(blue("JB Results:",['bold']))
print(list(zip(name, test)))
print("\n")

# Heteroskedasticity: Breush-Pagan --> Ho: var = constant
name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
test = sms.het_breuschpagan(ols_fit.resid, ols_fit.model.exog)
print(blue("BP Results:",['bold']))
print(list(zip(name, test)))

# Test for normality of residuals - JB test
stats.jarque_bera(ols_fit.resid)

"""Based on the output for the test for linearity, since the p-value is > alpha 0.05 we fail to reject the null and conclude that the model is linear

As for normality test, the JB Results show that the errors are normally distributed because the p-value is > alpha 0.05 even though the JB score = 1.8 which is > 1 (kurtosis is 3.2..., skewness is 0.2)

For Heteroskedasticity we observe that it is not possible to reject the null hypothesis that the variance is constant. This is due to the high p-value, and as such we can be fairly confident that as there is an increase in x there may be no observable increase in the variance of y.
"""

# Diagnostic plots of ols fit
figD = sm.graphics.plot_regress_exog(ols_fit, "MOT")
figD.set_figheight(10)
figD.set_figwidth(8)
plt.show()

"""MOV vs MOT shows a positive correlation. MOT vs residuals appears to be scattered around 0"""

model = smf.ols(formula='(MOV) ~ MOT + O1stD + BCORY + D1stD + BCDRY + Home_Games', data=eagles)
results = model.fit()
y = (eagles['MOV'])
y_pred=results.fittedvalues

# Using subset size =1
mr_sub = smf.ols(formula='(MOV) ~ MOT', data=eagles)
mr_sub_fit = mr_sub.fit()
y_sub=mr_sub_fit.fittedvalues

k = 8 # number of parameters in orginal model (includes y-intercept)
p = 3 # number of parameters in the subset model (includes y-intercept)

mallow.mallow(y, y_pred,y_sub, k, p)

model = smf.ols(formula='(MOV) ~ MOT + O1stD + BCORY + D1stD + BCDRY + Home_Games', data=eagles)
results = model.fit()
y = (eagles['MOV'])
y_pred=results.fittedvalues

# Using subset size =1
mr_sub = smf.ols(formula='(MOV) ~ D1stD', data=eagles)
mr_sub_fit = mr_sub.fit()
y_sub=mr_sub_fit.fittedvalues

k = 8 # number of parameters in orginal model (includes y-intercept)
p = 3 # number of parameters in the subset model (includes y-intercept)

mallow.mallow(y, y_pred,y_sub, k, p)

"""Using the results of the Mallow's CP the output showed that the best model with one predictor was MOV ~ MOT, followed by MOV~ D1stD. We tested the two models and found the first one had the lowest result, so we chose that as the best model with one predictor.

##Look at Cook’s distance Plot, Residuals Plot
"""

# Specify model
Turnover = smf.ols(formula='MOV ~ MOT', data=eagles)
T_results = Turnover.fit(cov_type='HC1')
print(T_results.summary())
T_results.resid.mean()

# Plot of residuals
sns.residplot(x='MOT', y='MOV', data=eagles,
              lowess=True, line_kws={'color':'red', 'lw':2, 'alpha':0.6})
plt.xlabel('Fitted Values')
plt.title('Residuals Plot')
plt.show

"""We can find that towards the end of the graph we find signficant high leverage observations in which the MOT shows a negative value but the MOV still remains quite high. These negative observations seem to heavily affect the residuals of the regression, even though it can be expected that a positive turnover margin would result a postive margin of victory."""

# QQ plot on studentized residuals
student_resid=T_results.get_influence().resid_studentized
df=len(student_resid)-4
t_dist=stats.t(df)
sm.qqplot(student_resid, line='45', dist=t_dist)
plt.grid()

"""In this QQ plot of the studentized residuals we see 7 observations that fall outside an absolute standard deviation of 2 from the mean."""

# Leverage plot
leverage=T_results.get_influence().hat_matrix_diag
plt.figure(figsize=(12,6))
plt.scatter(eagles.index,leverage)
plt.axhline(0,color='red')
plt.vlines(x=eagles.index, ymin=0,ymax=leverage)
plt.xlabel('Index')
plt.ylabel('Leverage')
plt.title('Diagnostic plot')
plt.show()

# Cook's distance plot
cooks_distance=T_results.get_influence().cooks_distance
plt.figure(figsize=(12,6))
plt.scatter(eagles.index,cooks_distance[0])
plt.axhline(0,color='red')
plt.vlines(x=eagles.index, ymin=0,ymax=cooks_distance[0])
plt.xlabel('Index')
plt.ylabel('Cook\'s Distance')
plt.title('Diagnostic plot')
plt.show()

"""The cook's distance graph shows one influential data point."""

fig, ax= plt.subplots(figsize=(10,5))
fig=sm.graphics.influence_plot(T_results, ax=ax, criterion='cooks')

"""## Evaluate the robustness of your coefficient estimates by bootstrapping your model. Provide a histogram of the bootstrapped estimates (including R^2), and comment on the findings. In particular how do these estimates compare against your LS estimates?"""

from sklearn.linear_model import LinearRegression

coefs=pd.DataFrame(columns=['B0','B1'])

for i in range(1000):
  sample = eagles.sample(eagles.shape[0], replace=True)
  results=smf.ols('MOV~MOT', sample).fit()
  b0,b1=results.params
  coefs=coefs._append({'B0':b0, 'B1':b1}, ignore_index=True)
# 5% CI
b0_u, b1_u = coefs.quantile(.975)
b0_l, b1_l = coefs.quantile(.025)

# Bootrstapped Histograms with CI
coefs.B0.hist()
plt.xlabel("Bootstrap Intercepts")
plt.ylabel("Frequency")
plt.axvline(b0_u, color = "red")
plt.axvline(b0_l, color = "red")

coefs.B1.hist()
plt.xlabel("Bootstrap B1")
plt.ylabel('Frequency')
plt.axvline(b1_u, color = 'red')
plt.axvline(b1_l, color = 'red')

"""Based on the bootstrap results the intercepts (about 2.8) and the slopes (about 4.5) are close enough to those from the results of tne LS estimates (2.77 and 4.49)

## Confidence intervals
"""

from scipy.stats import bootstrap

# Defining MOT coefficient
def reg_boot_b1(x,y):

    x = x.reshape((len(x),1))
    y = y.reshape((len(x),1))
    reg = LinearRegression().fit(x,y)

    return reg.coef_[0][0]
    #Pulling out Beta1 value

# Defining intercept
def reg_boot_intercept(x,y):

    x = x.reshape((len(x),1))
    y = y.reshape((len(x),1))
    reg = LinearRegression().fit(x,y)

    return reg.intercept_[0]
    # Pulling out Intercept value

X = eagles.MOT
Y = eagles.MOV
res = bootstrap((X,Y), reg_boot_b1, confidence_level=0.95, vectorized=False, method='BCa',
              paired=True)
print(res.confidence_interval)

"""With a 95% CI we are confident that the estimated reg_boot_b1 (beta 1) falls between (3.540241015898647, 5.526534689378149)"""

res = bootstrap((X,Y), reg_boot_intercept, confidence_level=0.95, vectorized=False, method='BCa',
              paired=True)
print(res.confidence_interval)

"""With a 95% CI we are confident that the estimated reg_boot_intercept (beta 0) falls between (1.0055391051206657, 4.604349878033104)

## Use cross-validation to evaluate your model’s performance
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

x = eagles[['MOT']]
y = eagles[['MOV']]
# Perform an OLS fit using all the data
regr = LinearRegression()
model = regr.fit(x,y)
regr.coef_
regr.intercept_

# Split the data into train  (70%)/test(30%) samples:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Train the model:
regr = LinearRegression()
regr.fit(x_train, y_train)

# Make predictions based on the test sample
y_pred = regr.predict(x_test)

# Evaluate Performance
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Perform a 5-fold CV
regr = linear_model.LinearRegression()
scores = cross_val_score(regr, x, y, cv=5, scoring='neg_root_mean_squared_error')
print('5-Fold CV RMSE Scores:', scores)

"""Root Mean Squared Error (RMSE), which measures the average prediction error made by the model in predicting the outcome for an observation shows a value of 12.765. For the variable MOT, the mean is 2.9, the median is 3, and Std is 14.

The lower the RMSE, the better the model. In this case we have a high RMSE which may mean that our model does not accurately explain all the dynamics that result in values we see in the MOV column. Our R^2 of 0.300 with an adjusted R^2 of 0.296 shows that initially regressing MOV on MOT only accounted for about 30% of the observed dynamics between the two variables.

Ultimately, we see that our value of RMSE falls just under the standard deviation and as a result we could be anywhere from 10-13 points off from the actual MOV. That is a value of 1.5 touchdowns to almost 2 touchdowns, which is quite large in terms of football as teams scores an average of 3.5 touchdowns per game in the modern NFL. As such it shows that we need more than one explanatory variable when it comes to truly analyzing MOV.

With that being said, ball possession and security are considered tantamount to success in the NFL. When the Eagles went on their 8-game winning streak in 2022, many highlighted the large positive turnover margin as a key to their success. But one could also see that the Eagles offense was excellent in terms of Red Zone Offense and large passing plays. These dynamics can't be shown, and also, it's questionable whether they would be necessary in a proper regression analysis due to their relation with the MOV variable and the ensuing endogeneity.

Overall, MOT is statistically significant when it comes to determining game outcomes in the form of MOV. But, using one variable is simply not enough to be able to predict something so complex and variable, especially when there are aspects of "relative" luck, skill, psychology, time of games, offensive prowess etc. MOT is a variable that we can use as a step to forming a more accurate multivariable regression, but can't explain everything that gives the resulting MOV.

So, while the RMSE values in comparison to our data may make it seem that using MOT to predict MOV may seem dubious, the real issue comes from the fact that the model itself is not robust due the lack of predictors. MOT is both valuable as a way to predict MOV and overall team success in both layman’s and a statistician’s terms, but it is just one of many confounding variables that determine on field performance in the game of football. You keep turnovers down on the offensive side of the ball, and get turnovers on the defensive side of the ball the more likely it is you are going to score and beat your opponent. The more you do this the greater the expected MOV, but this is not always the case, and that makes America’s sport so enigmatic in nature.
"""

