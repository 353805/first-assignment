import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
df_schema = pd.read_csv('survey_results_schema.csv')
df_survey = pd.read_csv('survey_results_public.csv',
                        usecols=['Respondent','YearsCode','Age1stCode','WorkWeekHrs','CodeRevHrs','Age'],
                        index_col='Respondent')
df_survey.head(20)
df_survey.shape
df_survey.dropna(inplace=True)
df_survey.shape
df_survey.dtypes
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
