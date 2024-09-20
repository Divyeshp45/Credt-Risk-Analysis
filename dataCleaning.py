import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Display settings
pd.set_option('display.max_columns', 125)
pd.set_option('display.max_rows', 125)

# Load the dataset
# Replace 'your_file.csv' with the actual file path of your dataset
app_df_1 = pd.read_csv('application_data.csv')

# Define the columns to drop
drop_columns = ['FLAG_CONT_MOBILE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                'FLAG_PHONE', 'FLAG_EMAIL', 'HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START',
                'FLOORSMAX_AVG', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'FLOORSMAX_AVG', 
                'FLOORSMAX_MODE', 'FLOORSMAX_MEDI', 'TOTALAREA_MODE', 'EMERGENCYSTATE_MODE',
                'REGION_POPULATION_RELATIVE', 'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BEGINEXPLUATATION_MEDI',
                'YEARS_BEGINEXPLUATATION_MODE', 'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
                'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
                'LIVE_CITY_NOT_WORK_CITY', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
                'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
                'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                'FLAG_DOCUMENT_21']

app_df_2 = app_df_1.drop(columns=drop_columns, axis=1)

# Looking at columns with missing values
print(round(100.0 * app_df_2.isnull().sum() / len(app_df_2), 2).sort_values())

# Checking dataframe info
app_df_2.info()
app_df_2.to_csv('Cleaned_applicationdata.csv')

# Finding % of people with outstanding dues and no outstanding dues.
target_0_percentage = round(len(app_df_2.query('TARGET==0')) / len(app_df_2), 4) * 100
print("Target_0_percentage:", target_0_percentage, "%")

target_1_percentage = round(len(app_df_2.query('TARGET==1')) / len(app_df_2), 4) * 100
print("Target_1_percentage:", target_1_percentage, "%")

# Creating Dataframe of non-defaulters and defaulters
target_0_df = app_df_2.query('TARGET==0')
target_1_df = app_df_2.query('TARGET==1')

print("Shape of Target 0 DataFrame:", target_0_df.shape)
print("Shape of Target 1 DataFrame:", target_1_df.shape)

# Checking unique values in each column
print(app_df_2.nunique().sort_values())

# Checking column types
print(app_df_2.dtypes)

# List of all categorical columns
categorical_columns = ['NAME_CONTRACT_TYPE', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CODE_GENDER',
                       'NAME_EDUCATION_TYPE', 'AMT_CATEGORY', 'AGE_GROUP', 'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT',
                       'AMT_REQ_CREDIT_BUREAU_HOUR', 'DEF_60_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                       'AMT_REQ_CREDIT_BUREAU_DAY', 'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_QRT',
                       'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AMT_REQ_CREDIT_BUREAU_MON', 
                       'AMT_REQ_CREDIT_BUREAU_YEAR', 'OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']

# List of all continuous numerical columns
numerical_columns = ['AMT_GOODS_PRICE', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 
                     'AMT_INCOME_TOTAL', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_BIRTH',
                     'AMT_CREDIT', 'AMT_ANNUITY']

# Loop for performing univariate analysis
for i in categorical_columns:
    
    if i in target_0_df.columns:
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        target_0_df[i].value_counts(normalize=True).plot.bar()
        plt.title(i + '- Target = 0')
        plt.subplot(1, 2, 2)
        target_1_df[i].value_counts(normalize=True).plot.bar()
        plt.title(i + '- Target = 1')
        plt.show()

# Correlation analysis for the entire Target data
plt.figure(figsize=(20, 10))
sns.heatmap(app_df_2[numerical_columns].corr(), annot=True)
plt.yticks(rotation=0)
plt.show()

# Correlation analysis for Target = 0
plt.figure(figsize=(20, 10))
sns.heatmap(target_0_df[numerical_columns].corr(), annot=True)
plt.yticks(rotation=0)
plt.show()

# Correlation analysis for Target = 1
plt.figure(figsize=(20, 10))
sns.heatmap(target_1_df[numerical_columns].corr(), annot=True)
plt.yticks(rotation=0)
plt.show()


# Select only numeric columns for correlation analysis
numeric_columns = target_0_df.select_dtypes(include=[np.number]).columns

# Correlation for numerical columns for Target = 0
corr_0 = target_0_df[numeric_columns].corr().where(np.triu(np.ones(target_0_df[numeric_columns].corr().shape), k=1).astype(bool))
corrdf0 = corr_0.unstack().reset_index()
corrdf0.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf0.dropna(subset=['Correlation'], inplace=True)
corrdf0['Correlation'] = round(corrdf0['Correlation'], 2)
corrdf0['Correlation_abs'] = corrdf0['Correlation'].abs()
print(corrdf0.sort_values(by='Correlation_abs', ascending=False).head(10))

# Correlation for numerical columns for Target = 1
corr_1 = target_1_df[numeric_columns].corr().where(np.triu(np.ones(target_1_df[numeric_columns].corr().shape), k=1).astype(bool))
corrdf1 = corr_1.unstack().reset_index()
corrdf1.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf1.dropna(subset=['Correlation'], inplace=True)
corrdf1['Correlation'] = round(corrdf1['Correlation'], 2)
corrdf1['Correlation_abs'] = corrdf1['Correlation'].abs()
print(corrdf1.sort_values(by='Correlation_abs', ascending=False).head(10))

'''   Shows error here:
# Correlation for numerical columns for Target = 0
corr_0 = target_0_df.corr().where(np.triu(np.ones(target_0_df.corr().shape), k=1).astype(np.bool))
corrdf0 = corr_0.unstack().reset_index()
corrdf0.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf0.dropna(subset=['Correlation'], inplace=True)
corrdf0['Correlation'] = round(corrdf0['Correlation'], 2)
corrdf0['Correlation_abs'] = corrdf0['Correlation'].abs()
print(corrdf0.sort_values(by='Correlation_abs', ascending=False).head(10))

# Correlation for numerical columns for Target = 1
corr_1 = target_1_df.corr().where(np.triu(np.ones(target_1_df.corr().shape), k=1).astype(np.bool))
corrdf1 = corr_1.unstack().reset_index()
corrdf1.columns = ['VAR1', 'VAR2', 'Correlation']
corrdf1.dropna(subset=['Correlation'], inplace=True)
corrdf1['Correlation'] = round(corrdf1['Correlation'], 2)
corrdf1['Correlation_abs'] = corrdf1['Correlation'].abs()
print(corrdf1.sort_values(by='Correlation_abs', ascending=False).head(10))
'''
# Boxplot analysis
for i in numerical_columns:
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    sns.boxplot(target_0_df[i])
    plt.title(i + '- Target = 0')
    plt.subplot(1, 2, 2)
    sns.boxplot(target_1_df[i])
    plt.title(i + '- Target = 1')
    plt.show()

# Multivariate analysis
pivot_data_1 = pd.pivot_table(app_df_2, values='TARGET', index=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], columns='NAME_INCOME_TYPE')
sns.heatmap(pivot_data_1, annot=True, cmap="Greens")
plt.yticks(rotation=0)
plt.show()

# Check if 'AMT_CATEGORY' exists in the DataFrame
if 'AMT_CATEGORY' in app_df_2.columns:
    pivot_data_2 = pd.pivot_table(app_df_2, values='TARGET', index=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], columns='AMT_CATEGORY')
    sns.heatmap(pivot_data_2, annot=True, cmap="Greens")
    plt.yticks(rotation=0)
    plt.show()
else:
    print("Column 'AMT_CATEGORY' not found in the DataFrame.")

'''
pivot_data_2 = pd.pivot_table(app_df_2, values='TARGET', index=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], columns='AMT_CATEGORY')
sns.heatmap(pivot_data_2, annot=True, cmap="Greens")
plt.yticks(rotation=0)
plt.show()  '''

pivot_data_3 = pd.pivot_table(app_df_2, values='TARGET', index=['CODE_GENDER', 'NAME_EDUCATION_TYPE'], columns='NAME_FAMILY_STATUS')
sns.heatmap(pivot_data_3, annot=True, cmap="Greens")
plt.yticks(rotation=0)
plt.show()

# Loop for bivariate analysis for numerical variables
for i in numerical_columns:
    sns.boxplot(data=app_df_2, x='TARGET', y=i)
    plt.show()

# Reading the data in pandas
prev_app_location = "previous_application.csv"
prev_app = pd.read_csv(prev_app_location)
prev_app.head()

# Checking previous application data info
prev_app.info()

# Checking column-wise missing values in previous application
print(round(100.0 * prev_app.isnull().sum() / len(prev_app), 2).sort_values())

# Dropping columns with more than 50% null values
prev_app.dropna(axis=1, thresh=int(prev_app.shape[0] * 0.5), inplace=True)

# Checking dataframe info after dropping
prev_app.info()

prev_app.to_csv("Cleaned_prev_applicationdata.csv")
