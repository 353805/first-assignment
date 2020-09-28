import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from scipy import stats
import math
from sklearn.preprocessing import OneHotEncoder
df_schema = pd.read_csv('survey_results_schema.csv')
df_survey = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent','YearsCode','Age1stCode','WorkWeekHrs','CodeRevHrs','Age'],
                        index_col='Respondent')
df_survey.head(20)
df_survey.shape
df_survey.dropna(inplace=True)
df_survey.shape
df_survey['CodeRevHrs'] = df_survey['CodeRevHrs'].astype('int64')
df_survey.info()
df_survey.replace(to_replace='Younger than 5 years', value='4', inplace=True)
df_survey.replace(to_replace='Older than 85', value='86', inplace=True)
df_survey.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)
column_values = df_survey[['WorkWeekHrs']].values.ravel()
unique_values = pd.unique(column_values)
column_values2 = df_survey[['CodeRevHrs']].values.ravel()
unique_values2 = pd.unique(column_values2)
df_survey = df_survey.astype('int64', copy=False)
df_survey.info()
plt.plot(df_survey['Age'], df_survey['YearsCode'], 'ro', markersize=0.3)
plt.xlabel('Age')
plt.ylabel('WorkWeekHrs')
plt.plot(df_survey['CodeRevHrs'], df_survey['Age1stCode'], 'ro', markersize=0.3)
plt.xlabel('CodeRevHrs')
plt.ylabel('Age1stCode')
plt.plot(df_survey['Age'], df_survey['Age1stCode'], 'ro', markersize=0.3)
plt.xlabel('Age')
plt.ylabel('Age1stCode')
plt.plot(df_survey['CodeRevHrs'], df_survey['WorkWeekHrs'], 'ro', markersize=0.3)
plt.xlabel('CodeRevHrs')
plt.ylabel('WorkWeekHrs')
plt.show()
print (df_survey.corr())
print (df_survey.describe())
#x1 = Age
#x2 = Age1stCode
#y = YearsCode
df_survey ['CodeRev'] = pd.read_csv('survey_results_public.csv',
                        usecols=['CodeRev'])
#column_values3 = df_survey[['CodeRev']].values.ravel()
#unique_values3 = pd.unique(column_values3)
df_survey.replace(to_replace='Yes, because I see value in code review', value='Yes', inplace=True)
df_survey.replace(to_replace='Yes, because I was told to do so', value='Yes', inplace=True)
CodeRev = df_survey['CodeRev']
for index, category in CodeRev.items():
    if (not isinstance(category, str) and math.isnan(category)):
        CodeRev[index] = 0
df_survey['CodeRev'] = CodeRev
df_survey.replace(to_replace='Yes', value='1', inplace=True)
df_survey.replace(to_replace='No', value='0', inplace=True)
df_survey ['Student'] = pd.read_csv('survey_results_public.csv',
                        usecols=['Student'])
#column_values4 = df_survey[['Student']].values.ravel()
#unique_values4 = pd.unique(column_values4)
X = df_survey['Student']
X_new = []
unique_categories = {}
current_index = 0
for index, value in X.items():
    category = value
    if (not isinstance(category, str) and math.isnan(category)):
        category = 'No answer'

    if (category not in unique_categories):
        current_index += 1
        unique_categories[category] = current_index
        X_new.append([current_index, category])
    X[index] = unique_categories[category]

ohe = OneHotEncoder(categories='auto')
students_categories = ohe.fit_transform(X_new).toarray()
df_survey['CodeRev'] = df_survey['CodeRev'].astype('int64')
df_survey['Student'] = X.astype('int64')
#print (df_survey)
df_survey_clr = df_survey[['YearsCode','Age1stCode','Age','CodeRev','Student']]
#df_survey_clr.isna().sum()
#df_survey_clr.corr()
Q1 = df_survey_clr.quantile(0.25)
Q3 = df_survey_clr.quantile(0.75)
IQR = Q3 - Q1
df_survey_clr_q = df_survey_clr[~((df_survey_clr < (Q1 - 1.5 * IQR)) | (df_survey_clr > (Q3 + 1.5 * IQR))).any(axis=1)]
df_survey_clr_sd = df_survey_clr[np.abs(df_survey_clr - df_survey_clr.mean()) <= 3*df_survey_clr.std()]
df_survey_clr_sd.isna().sum()
df_survey_clr_sd = df_survey_clr_sd.dropna()
#print(df_survey_clr.corr())
#print(df_survey_clr_sd.corr())
print(df_survey_clr_sd.YearsCode.mean())
print(df_survey_clr_sd.YearsCode.max())
print(df_survey_clr_sd.YearsCode.min())
print(df_survey_clr_sd.YearsCode.sum())
print(df_survey_clr.YearsCode.mean())
print(df_survey_clr.YearsCode.max())
print(df_survey_clr.YearsCode.min())
print(df_survey_clr.YearsCode.sum())
df_survey_clr_m = df_survey_clr[df_survey_clr.YearsCode < 40]
x = df_survey_clr['YearsCode']
plt.hist(x, bins=100)
plt.show();
print(df_survey_clr[df_survey_clr.YearsCode > 40].describe())
print(df_survey_clr.describe())
sns.boxplot(y='YearsCode', data=df_survey_clr)
plt.show();
sns.boxplot(y='YearsCode', data=df_survey_clr_q)
plt.show();
sns.boxplot(y='YearsCode', data=df_survey_clr_sd)
plt.show();
sns.boxplot(y='YearsCode', data=df_survey_clr_m)
plt.show();
df_survey_clr_row = df_survey_clr[df_survey_clr.YearsCode > 40].index
df_survey_clr_fin = df_survey_clr.drop(df_survey_clr_row, axis=0)
print(df_survey_clr_fin.describe())
reg4_1 = linear_model.LinearRegression()
reg4_1_r = reg4_1.fit(df_survey_clr_fin[['Age']], df_survey_clr_fin['YearsCode'])
#reg4_1_y = df_survey_clr_fin['Age']
reg4_1_y_p = reg4_1.predict([[60]])
#mse4_1 = mean_squared_error(reg4_1_y, reg4_1_y_p.tolist())
print(reg4_1_y_p)
reg4_2 = linear_model.LinearRegression()
reg4_2_r = reg4_2.fit(df_survey_clr_fin[['Age','Age1stCode']], df_survey_clr_fin['YearsCode'])
#reg4_2_y = df_survey_clr_fin['Age']
reg4_2_y_p = reg4_2.predict(df_survey_clr_fin[['Age','Age1stCode']])
#mse4_2 = mean_squared_error(reg4_2_y, reg4_2_y_p)
print(reg4_2_y_p)
reg4_3 = linear_model.LinearRegression()
reg4_3_y = reg4_3.fit(df_survey_clr_fin[['Age','Age1stCode','CodeRev','Student']],df_survey_clr_fin['YearsCode'])
reg4_3_y_p = reg4_3.predict(df_survey_clr_fin[['Age','Age1stCode','CodeRev','Student']])
print(reg4_3_y_p)
