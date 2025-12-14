import streamlit as st
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
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
st.write('As the target variable CVD was entered retroactively in the dataset'
' (if a person developed CVD later in life, all previous entries were marked as having CVD.' \
' Therefore, our target variable was created using the following logic:')
"""
- If a person developed CVD within 6 years, they are marked as developing CVD early
- If a person developed CVD within 24 years (study duration), they are marked as developing CVD late.
- If a person did not develop CVD within the study period, they are marked as never developing CVD.
"""
### TO SEE THE RELEVANT CODE PLEASE CHECK AFTER THE PREPROCESSING PART IS FINISHED


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
st.caption('Table 1: Summary of missingness per variable in descending order.')

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

st.write('All erroneous datapoints were winsorised to the closest biologically plausible values as indicated'
' by the threshholds described above.' \
' It should be noted that these values are still quite extreme and may have underlying issues in measurement,' \
' protocol or other. ')

#Winsorising
#Utilising threshholds described in text
cvd_imputed.loc[cvd_imputed['SYSBP'] > 250, 'SYSBP'] = 250
cvd_imputed.loc[cvd_imputed['DIABP'] < 40, 'DIABP'] = 40
cvd_imputed.loc[cvd_imputed['DIABP'] > 120, 'DIABP'] = 120
cvd_imputed.loc[cvd_imputed['TOTCHOL'] > 500, 'TOTCHOL'] = 500
cvd_imputed.loc[cvd_imputed['GLUCOSE'] > 400, 'GLUCOSE'] = 400
cvd_imputed.loc[cvd_imputed['BMI'] > 55, 'BMI'] = 55
cvd_imputed.loc[cvd_imputed['HEARTRTE'] > 200, 'HEARTRTE'] = 200
cvd_imputed.loc[cvd_imputed['CIGPDAY']> 80, 'CIGPDAY'] = 80



st.subheader('Outliers and normality checks')
st.write('As we are working with biological data, our data will initially show many outliers' \
' when using outlier metrics like IQR or Z-score. This is due to how biological data tends to be' \
' right-skewed, especially when including diseased populations (i.e. hypertension).')
st.write('Below is an example of this phenomenon using systolic blood pressure:')

#making the figure

fig, ax = plt.subplots(figsize=(10, 4))
ax.boxplot(cvd_imputed['SYSBP'], vert=False)
ax.set_title('Boxplot of Systolic Blood Pressure')
ax.set_xlabel('Systolic Blood Pressure (mmHg)')
st.pyplot(fig)
st.caption('Figure 1: Boxplot of Systolic Blood Pressure. Circles indicate datapoints greater than 1.5*IQR')
"""
**Why is this important?**
"""
st.write('Having skewed data means our data is non-normal. Normality is a core assumption in many statistical' \
' tests, as well as in many machine learning models (especially those utilising regression). Using non-normal' \
' data may therefore decrease performance and/or lead to incorrect statistical conclusions.')
st.write('To mediate this issue, we applied log-transformations to all continuous variables determined' \
' to be right-skewed. Skewness was determined using Shapiro-Wilk normality tests.')
st.write('Below is a summary of the normality tests, including variable name, p-value and type of log' \
' transformation applied (log, log1p (handles 0)).')
#Normalisation functions
def normalize(df): # Only to be run on non-normal columns
    data_nona = df.dropna()
    skewness = stats.skew(data_nona)
    df_normalized = df.copy()
    if skewness > 0:  # Right-skewed (0 would mean no skew, lower than 0 is left skew)
        #Log transformation (with handling of zero-values)
        if (data_nona > 0).all():
            df_normalized = np.log(df)
            st.write(f"      -> Applied log transformation")
        else:
            df_normalized = np.log1p(df)
            st.write(f"      -> Applied log1p transformation")

    return df_normalized

def normality_check_and_fix(df):
    df_normalized = df.copy()
    print("Shapiro-Wilk Normality Test")

    for column in df.columns:
      if len(df[column].unique()) > 5: #Just checks that it's not a categorical variable:
        data_nona = df[column].dropna()
        stat, p_value = stats.shapiro(data_nona)

        if p_value > 0.05:
            st.write(f"{column}: Normal (p={p_value:.4f})")
        else:
            st.write(f"{column}: Not Normal (p={p_value:.4f})")
            df_normalized[column] = normalize(df[column])
    #print(Style.RESET_ALL)
    return df_normalized;

cols_norm_test = [
    "AGE", "SEX", "SYSBP", "DIABP", "TOTCHOL", "BMI",
    "CURSMOKE", "CIGPDAY", "DIABETES", "GLUCOSE",
    "BPMEDS", "PREVHYP", "HEARTRTE", "CVD"
]
cvd_imputed_normalized = normality_check_and_fix(cvd_imputed[cols_norm_test])
st.caption('Summary of normality tests results. Abbreviations are: SYSBP = systolic blood pressure, DIABP = diastolic blood pressure,' \
' TOTCHOL = total cholesterol, CIGPDAY = cigarettes smoked per day, GLUCOSE = glucose levels, HEARTRTE = heart rate.')


# Interactive module showing difference between original and normalised data
log_vars = ['SYSBP','DIABP','TOTCHOL','GLUCOSE','BMI','CIGPDAY','HEARTRTE']


def log_transform_visualization(variable):

    data = cvd_imputed[variable].dropna()
    transformed = cvd_imputed_normalized[variable].dropna()

    # decide whether log or log1p was used
    if (data > 0).all():
      transform_name = "log"
    else:
      transform_name = "log1p"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # Original distribution
    #plt.subplot(1,2,1)
    sns.histplot(data, bins=30, kde=True, color='plum', edgecolor='purple', ax=ax1)
    ax1.set_title(f"Original Distribution of {variable}")
    ax1.set_xlabel(variable)
    ax1.set_ylabel("Frequency")

    # Transformed distribution
    #plt.subplot(1,2,2)
    sns.histplot(transformed, bins=30, kde=True, color='lightblue', edgecolor='blue', ax=ax2)
    ax2.set_title(f"{transform_name} Transformed Distribution of {variable}")
    ax2.set_xlabel(f"{transform_name}({variable})")
    ax2.set_ylabel("Frequency")

    return fig


#Dropdown menu
dropdown = st.selectbox('Select', ['SYSBP','DIABP','TOTCHOL','GLUCOSE','BMI','CIGPDAY','HEARTRTE'])
st.pyplot(log_transform_visualization(dropdown)) #log_transform_visualization(dropdown)

st.caption('Figure 2: Comparison of variable distribution before and after log-transformation.')


### DEFINING OUTCOME VARIABLE

cvd_imputed_normalized['CVD_MULTI'] = 1
cvd_imputed_normalized.loc[cvd_imputed.TIMECVD > 2190, 'CVD_MULTI'] = 2
cvd_imputed_normalized.loc[cvd_imputed.TIMECVD == 8766, 'CVD_MULTI'] = 0


st.title('Population characteristics and outcome variable description')

st.subheader('Outcome variable proportions')
st.write('As stated in our data preperation, our outcome variable ahs the following subcategories:')
"""
- 0 = never developed CVD (within study duration)
- 1 = develoepd CVD early (within 6 years of baseline measurement)
- 2 = developed CVD later (within study length of 24 years)"""
# Proportions of our outcome variable
fig, ax = plt.subplots()
counts = cvd_imputed['CVD_MULTI'].value_counts().sort_index() / cvd_imputed_normalized.shape[0]
counts.plot(kind='bar', ax=ax, color=['lightgreen', 'lightblue', 'plum'])
ax.set_xlabel('CVD Development Category')

st.pyplot(fig)

st.caption('Figure 3: Proportions of outcome variable categories. 0 = never developed CVD, 1 = developed CVD early (within 6 years)' \
', 2 = developed CVD late (within 24 years).')

st.subheader('Population characteristics')
st.write('Below are a summary of basic descriptive statistics. Please note that all descriptive statistics are taken on the imputed, but not log-transformed dataset')


st.title('References')
st.write(' 1. World Health Organisation (2025), "Cardiovascular diseases (CVDs)", pls check how to do manual reference https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)')


