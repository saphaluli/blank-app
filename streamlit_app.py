import streamlit as st
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levene
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, precision_recall_curve

st.title("🤍 Predicting onset of Cardiovascular disease from data?")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

st.title('Introduction')
st.subheader('Cardiovascular disease and research aim')
#first paraphraph
st.write('Cardiovascular disease (CVD) is the leading cause of death globally, accounting for' \
' approximately 38\% of global deaths (1, WHO (2025). The term CVD includes a range of diseases, such as stroke,' \
' heart attack or peripheral arterial disease. CVD is often caused by a combination of unhealthy' \
' lifestyle factors. Early detection of patients with high risk of developing CVD is therefore very' \
' important, as modifying lifestyle factors can greatly decrease the risk of developing a disease under the CVD umbrella.'
)
#Second paragraph - research aim/description of model
st.write('Through the use of data analysis and different AI/machine learning approaches, this project' \
'aims to quantify which baseline factors are associated with earlier developments of CVD, and whether this ' \
' risk can be predicted accurately through these factors.')

st.title('Methodology: Preperation of our data')
st.subheader('Data source')
st.write('The data used for our analysis and machine learning models are sourced from the Framingham Heart Study'
' (2, add reference to the datset). Add more stuff about the dataset, when, what is measured etc.')

# This is the second parapgraph about which variables we selected and why.
st.subheader('Data selection')
st.write('We selected the following features based on.... data leakage. ')
"""
- Age
- Sex (male, female)
- Systolic blood pressure
- Diastolic blood pressure
- Total cholesterol
- BMI
- Smoking status (smoker, non-smoker)
- Cigaretted smoked per day
- Diabetes (yes, no)
- Glucose levels
- Taking blood pressure medication (yes/no)
- Is hypertensive (yes/no)
- Heart rate
"""
st.write('As the target variable CVD was entered retroactively in the dataset (if a The target variable was created using the following logic:')

# 1. Load dataset from link
cvd = pd.read_csv('https://raw.githubusercontent.com/LUCE-Blockchain/Databases-for-teaching/refs/heads/main/Framingham%20Dataset.csv')
cols = [
    "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", "BMI",
    "CURSMOKE", "CIGPDAY", "DIABETES", "GLUCOSE",
    "BPMEDS", "PREVHYP", "HEARTRTE", "CVD", "TIMECVD"
    ] #PREVCHD, etc. not includes as would introduce data leakage since these are apart of CVD

##Relevant rows:

  #Consider how 'PERIOD' may change. What about dropouts? What if a person dies? Maybe only choose period 3 or 1?
  #Use period 1 because period 1 first of all has a bigger sample, and using period 3 could lead to data leakage
  #because the measurements might already be influenced by disease development. --> We need data before CVD.
  #Looking at other predictive models like the Framingham Risk Score: they also use baseline values

# Only select first period
cvd = cvd[cvd["PERIOD"] == 1]
st.title('Methodology: Checking missing and erroneous data')
st.subheader('Missing data')

st.write('In the table below is a summary of missing data per column, ranked in descending order.')
def check_missing(df, col):
    missing  = 0
    misVariables = []
    CheckNull = df.isnull().sum()
    for var in range(0, len(CheckNull)):
        misVariables.append([col[var], CheckNull[var], round(CheckNull[var]/len(df),3)])
        missing = missing + 1

    if missing == 0:
        st.write('Dataset is complete with no blanks.')
    else:
        df_misVariables = pd.DataFrame.from_records(misVariables)
        df_misVariables.columns = ['Variable', 'Missing', 'Percentage']
        s = df_misVariables.sort_values(by=['Percentage'], ascending=False).style.bar(subset=['Percentage'], color='#d65f5f')
        st.write(s)
    return df_misVariables

ranked_df_missing_value = check_missing(cvd[cols], cols)

st.write('We imputed the following variables: ')

### START OF IMPUTATIONNNN ###

## ON BPMEDS imputation
st.write('For the feature Blood pressure medication, we imputed missing values based on' \
' whether the patient was hypertensive or not.')
"""
- If the patient is not hypertensive -> impute as 0 (not taking medication)
- If the patient is hypertensive -> drop these patients (only small percentage missing, cannot know whether they took medication or not)"""

### Making new imputed df
cvd_imputed = cvd[cols].copy()

#From phase 1
# If NOT hypertensive -> impute BPMEDS with 0
cvd_imputed.loc[
    (cvd_imputed['PREVHYP'] == 0) & (cvd_imputed['BPMEDS'].isna()),
    'BPMEDS'
] = 0

# If hypertensive & BPMEDS missing -> drop (only small percentage missing)
cvd_imputed = cvd_imputed.dropna(subset=['BPMEDS'])

### Impute smoking status
st.write('A similar logic was used for missing in cigarettes smoked per day, where non-smokers with missing' \
' amount of cigerattes smoked per day were imputed as 0.')
#If someone is no smoker (0) and CIGPDAY is missing --> impute with 0
cvd_imputed.loc[
    (cvd_imputed['CURSMOKE'] == 0) & (cvd_imputed['CIGPDAY'].isna()),
    'CIGPDAY'
] = 0

### impute continuous variables

st.write('The method of imputations for continuous variables'
' (glucose levels, total cholesterol, BMI, heart rate, cigarettes smoked per day (if person was smoker)) was done using the median.')
#GLUCOSE, TOTCHOL, BMI, HEARTRTE -> impute with median
num_cols_median = ["GLUCOSE", "TOTCHOL", "BMI", "HEARTRTE"]

for col in num_cols_median:
    median_val = cvd_imputed[col].median()
    cvd_imputed[col] = cvd_imputed[col].fillna(median_val)

#If someone is a smoker (1) and CIGPDAY is missing --> impute with median from smokers
median_cigs_smokers = cvd_imputed.loc[
    cvd_imputed['CURSMOKE'] == 1, 'CIGPDAY'
].median()

cvd_imputed.loc[
    (cvd_imputed['CURSMOKE'] == 1) & (cvd_imputed['CIGPDAY'].isna()),
    'CIGPDAY'
] = median_cigs_smokers

#Throws error if imputation went wrong for some reason
assert cvd_imputed.isna().sum().sum() == 0 

st.subheader('Erroneous data')
st.write('Erroneous data was selected based on blah blah blah. Below is a summary of erroneous datapoints.' \
' The following threshholds were used:')
"""
- Systolic blood pressure (SYSBP): 70-250 mmHg
- Diastolic blood pressure (DIABP): 40-120 mmHg
- BMI: 10-55 kg/m²
- Glucose levels: 40-400 mg/dL
- Heart rate: 30-200 bpm
- Cigarettes smoked per day: 0-80"""
#Identify suspicous values
erroneous_checks = {
    "SYSBP": (70, 250),
    "DIABP": (40, 120),
    "BMI": (10, 55),
    "GLUCOSE": (40, 400),
    "HEARTRTE": (30, 200),
    "CIGPDAY": (0, 80),
}

for col, (low, high) in erroneous_checks.items():
    invalid = cvd_imputed[(cvd_imputed[col] < low) | (cvd_imputed[col] > high)]
    if len(invalid) > 0:
      st.write(f"{col}: has {len(invalid)} values not in accordance with biology.")



st.title('References')
st.write(' 1. World Health Organisation (2025), "Cardiovascular diseases (CVDs)", pls check how to do manual reference https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)')


