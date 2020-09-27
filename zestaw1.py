import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#1
col_names=['CENA', 'LICZBA_POKOI','METRAZ','liczba','ADRES','OPIS']
df = pd.read_csv("./train.tsv", sep="\t", names=col_names)
SredniaCena = df.loc[:,"CENA"].mean().round(0)
SredniaCenaDF = pd.DataFrame(data=[SredniaCena])
SredniaCenaDF.to_csv('out0.csv')
#2
df['CENA_ZA_METR'] = df['CENA']/df['METRAZ']
Out1_Sr = df.loc[:,"CENA_ZA_METR"].mean()
Out1_PrSr = df.loc[(df['CENA_ZA_METR']>Out1_Sr) & (df['LICZBA_POKOI']>=3)]
Out1_K = Out1_PrSr['CENA'],Out1_PrSr['LICZBA_POKOI'],Out1_PrSr['CENA_ZA_METR']
Out1PD = pd.DataFrame.from_records(Out1_K)
Out1PD.to_csv('out1.csv')
#3
df2 = pd.read_csv("./train.tsv", sep="\t", names=col_names)
col_names2 = ['PIETRO','opis_pietra']
df3 = pd.read_csv("./description.csv", sep=",")
df3.head()
df2 = pd.merge(df2, df3,how='left', on='liczba')
df2.to_csv('out2.csv')
#4
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', -1)
df_schema = pd.read_csv('survey_results_schema.csv')
df_survey = pd.read_csv('survey_results_public.csv',usecols=['Respondent','WorkWeekHrs','CodeRevHrs','Student'],index_col='Respondent')
df_survey.head(20)
df_survey.shape
df_survey.dropna(inplace=True)
df_survey.shape
df_survey.dtypes
df_survey['WorkWeekHrs'] = df_survey['WorkWeekHrs'].astype('int64')
df_survey['CodeRevHrs'] = df_survey['CodeRevHrs'].astype('int64')
#df_survey.info()
#print (df_survey)
#5
column_values = df_survey[['WorkWeekHrs']].values.ravel()
unique_values = pd.unique(column_values)
column_values2 = df_survey[['CodeRevHrs']].values.ravel()
unique_values2 = pd.unique(column_values2)
plt.plot(df_survey['CodeRevHrs'], df_survey['WorkWeekHrs'], 'ro', markersize=0.3)
plt.xlabel('CodeRevHrs')
plt.ylabel('WorkWeekHrs')
plt.show()
#df4_survey = pd.read_csv('survey_results_public.csv',usecols=['Respondent','Student'],index_col='Respondent')
#df4_survey.info()
column_values3 = df_survey[['Student']].values.ravel()
unique_values3 = pd.unique(column_values3)
df4_survey_1 = df_survey.loc[(df_survey['Student']=='No')]
df4_survey_2 = df_survey.loc[(df_survey['Student']=='Yes, full-time')]
df4_survey_3 = df_survey.loc[(df_survey['Student']=='Yes, part-time')]
print(df4_survey_3)




