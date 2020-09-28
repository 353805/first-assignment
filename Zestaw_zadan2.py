import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
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
df_survey.replace(to_replace='Younger than 5 years', value='4', inplace=True)
df_survey.replace(to_replace='Older than 85', value='86', inplace=True)
df_survey.replace(to_replace={'Less than 1 year': '0',
                              'More than 50 years': '51'},
                  inplace=True)
df_survey = df_survey.astype('int64', copy=False)
plt.plot(df_survey['Age'], df_survey['YearsCode'], 'ro', markersize=0.3)
plt.xlabel('Age')
plt.ylabel('YearsCode')
plt.show()
plt.plot(df_survey['Age'], df_survey['Age1stCode'], 'ro', markersize=0.3)
plt.xlabel('Age')
plt.ylabel('Age1stCode')
plt.show()
plt.plot(df_survey['Age1stCode'], df_survey['YearsCode'], 'ro', markersize=0.3)
plt.xlabel('Age1stCode')
plt.ylabel('YearsCode')
plt.show()
#x1 = Age
#x2 = Age1stCode
#y = YearsCode
df_survey ['CodeRev'] = pd.read_csv('survey_results_public.csv',
                        usecols=['CodeRev'])
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
df_survey_clr = df_survey[['YearsCode','Age1stCode','Age','CodeRev','Student']]
Q1 = df_survey_clr.quantile(0.25)
Q3 = df_survey_clr.quantile(0.75)
IQR = Q3 - Q1
df_survey_clr_q = df_survey_clr[~((df_survey_clr < (Q1 - 1.5 * IQR)) | (df_survey_clr > (Q3 + 1.5 * IQR))).any(axis=1)]
df_survey_clr_sd = df_survey_clr[np.abs(df_survey_clr - df_survey_clr.mean()) <= 3*df_survey_clr.std()]
df_survey_clr_sd.isna().sum()
df_survey_clr_sd = df_survey_clr_sd.dropna()
df_survey_clr_m = df_survey_clr[df_survey_clr.YearsCode < 40]
x = df_survey_clr['YearsCode']
plt.hist(x, bins=100)
plt.show();
df_survey_clr_row = df_survey_clr[df_survey_clr.YearsCode > 40].index
df_survey_clr_fin = df_survey_clr.drop(df_survey_clr_row, axis=0)
print(df_survey_clr.corr())
print(df_survey_clr_fin.corr())
sns.boxplot(y='YearsCode', data=df_survey_clr_fin)
plt.show();
reg4_1 = linear_model.LinearRegression()
reg4_1_r = reg4_1.fit(df_survey_clr_fin[['Age']], df_survey_clr_fin['YearsCode'])
reg4_1_y_p = reg4_1.predict([[60]])
reg4_2 = linear_model.LinearRegression()
reg4_2_r = reg4_2.fit(df_survey_clr_fin[['Age','Age1stCode']], df_survey_clr_fin['YearsCode'])
reg4_2_y_p = reg4_2.predict(df_survey_clr_fin[['Age','Age1stCode']])
reg4_3 = linear_model.LinearRegression()
reg4_3_y = reg4_3.fit(df_survey_clr_fin[['Age','Age1stCode','CodeRev','Student']],df_survey_clr_fin['YearsCode'])
reg4_3_y_p = reg4_3.predict(df_survey_clr_fin[['Age','Age1stCode','CodeRev','Student']])

