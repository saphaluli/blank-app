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
st.write(
    "The original CVD variable in the dataset is coded retrospectively, meaning that participants "
    "who developed cardiovascular disease at any point during follow-up were marked as having CVD "
    "for all earlier records. As a result, this variable does not distinguish between early and late "
    "disease onset. To better capture differences in timing of CVD development, a multiclass outcome "
    "variable was constructed using the following definition:"
)

st.markdown("""
- **0**: No CVD during the study follow-up  
- **1**: Early CVD onset (within 6 years of baseline)  
- **2**: Late CVD onset (after 6 years, up to 24 years)
""")

st.write(
    "Alternative thresholds were explored during model development, but a 6-year cutoff was retained "
    "as it provided a balance between class distribution and model stability."
)
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

cvd_imputed['CVD_MULTI'] = 1
cvd_imputed.loc[cvd_imputed.TIMECVD > 2190, 'CVD_MULTI'] = 2
cvd_imputed.loc[cvd_imputed.TIMECVD == 8766, 'CVD_MULTI'] = 0

#also add for cvd_imputed_normalized
cvd_imputed_normalized['CVD_MULTI'] = cvd_imputed['CVD_MULTI']

st.title('Population characteristics and outcome variable description')

st.subheader('Outcome variable proportions')
st.write('As stated in our data preperation, our outcome variable has the following subcategories:')
"""
- 0 = never developed CVD (within study duration)
- 1 = developed CVD early (within 6 years of baseline measurement)
- 2 = developed CVD later (within study length of 24 years)"""

# Proportions of our outcome variable
fig, ax = plt.subplots()
counts = cvd_imputed_normalized['CVD_MULTI'].value_counts().sort_index() / cvd_imputed.shape[0]
counts.plot(kind='bar', ax=ax, color=['lightgreen', 'lightblue', 'plum'])
ax.set_xlabel('CVD Development Category')

st.pyplot(fig)

st.caption('Figure 3: Proportions of outcome variable categories. 0 = never developed CVD, 1 = developed CVD early (within 6 years)' \
', 2 = developed CVD late (within 24 years).')

st.subheader('Population characteristics')
st.write('Below are a summary of basic descriptive statistics. Please note that all descriptive statistics are taken on the imputed, but not log-transformed dataset')

#TABEL 1 - Numerical descriptive statistics
#adding median as well
st.write('Table 2 summarizes the central tendency and dispersion of the main continuous variables in the cohort.') 
st.caption('Table2: Descriptive statistics for numerical variables.')
numeric_vars = ['AGE','SYSBP','DIABP','TOTCHOL','BMI','CIGPDAY','GLUCOSE','HEARTRTE','TIMECVD']
desc = cvd_imputed[numeric_vars].describe().T.drop(labels='count', axis=1) #Count stays the same so not needed
desc['median'] = cvd_imputed[numeric_vars].median()
st.write(desc)

st.caption('Table 3: Descriptive statistics for categorical variables.')
#TABEL 2 - Categorical descriptive statistics
def categorical_summary(df, column, labels=None):
    summary = df[column].value_counts(normalize=True).rename("Proportion") * 100
    summary = summary.reset_index()
    summary.columns = [column, "Percentage (%)"]

    if labels:
        summary[column] = summary[column].map(labels)

    summary["Percentage (%)"] = summary["Percentage (%)"].round(1)
    summary.columns = ['TYPE', "Percentage (%)"]
    return summary

sex_table = categorical_summary(
    cvd_imputed,
    "SEX",
    labels={1: "Male", 2: "Female"}
)

smoke_table = categorical_summary(
    cvd_imputed,
    "CURSMOKE",
    labels={0: "Non-smoker", 1: "Current smoker"}
)

diabetes_table = categorical_summary(
    cvd_imputed,
    "DIABETES",
    labels={0: "No diabetes", 1: "Diabetes"}
)

hypertension_table = categorical_summary(
    cvd_imputed,
    "PREVHYP",
    labels={0: "No hypertension", 1: "Hypertension"}
)

bpmeds_table = categorical_summary(
    cvd_imputed,
    "BPMEDS",
    labels={0: "No BP medication", 1: "BP medication"}
)

categorical_tables = {
    "Sex": sex_table,
    "Smoking status": smoke_table,
    "Diabetes": diabetes_table,
    "Hypertension": hypertension_table,
    "BP medication": bpmeds_table
}
##Trying to add all tables together, I know it's a bit messy but I'm adjusting other's code

full_table = pd.DataFrame()

for title, table in categorical_tables.items():
    full_table = pd.concat([full_table, table])

st.write(full_table)

st.subheader("Visual exploration")

st.write("The histogram below shows the age distribution of the study population.")

# Histogram AGE
st.markdown("**Age distribution**")
fig, ax = plt.subplots()
ax.hist(cvd_imputed["AGE"], bins=20)
ax.set_title("Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
st.pyplot(fig)
st.caption("Figure 4: Histogram of age in the cohort.")

st.write("The boxplot below compares systolic blood pressure across the three CVD outcome classes.") 

# Boxplot SYSBP by CVD class
st.markdown("**Systolic blood pressure across CVD classes**")
fig, ax = plt.subplots(figsize=(8, 6))
classes = sorted(cvd_imputed["CVD_MULTI"].unique())
groups = [cvd_imputed.loc[cvd_imputed["CVD_MULTI"] == c, "SYSBP"] for c in classes]
ax.boxplot(groups, labels=classes)
ax.set_title("SYSBP Across CVD Classes")
ax.set_xlabel("CVD Class")
ax.set_ylabel("SYSBP (mmHg)")
st.pyplot(fig)
st.caption("Figure 5: Boxplot of systolic blood pressure by CVD_MULTI class.")

st.write("Age distributions are compared across CVD outcome classes to illustrate age-related differences.")

# Boxplot AGE by CVD class
st.markdown("**Age across CVD classes**")
fig, ax = plt.subplots(figsize=(8, 6))
groups_age = [cvd_imputed.loc[cvd_imputed["CVD_MULTI"] == c, "AGE"] for c in classes]
ax.boxplot(groups_age, labels=classes)
ax.set_title("Age Across CVD Classes")
ax.set_xlabel("CVD Class")
ax.set_ylabel("Age (years)")
st.pyplot(fig)
st.caption("Figure 6: Boxplot of age by CVD_MULTI class.")


st.write("The interactive plot below allows users to explore how the distribution of CVD changes when restricting the cohort to older individuals.")

min_age = st.slider("Minimum age",
                    int(cvd_imputed["AGE"].min()),
                    int(cvd_imputed["AGE"].max()),
                    50)

filtered = cvd_imputed[cvd_imputed["AGE"] >= min_age]

fig, ax = plt.subplots()
filtered["CVD_MULTI"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title(f"CVD_MULTI distribution for AGE ≥ {min_age}")
st.pyplot(fig)
st.caption("Figure 7: Age slider per CVD classes")

# Select relevant variables for correlation analysis
corr_vars = [
    "AGE",
    "SYSBP",
    "DIABP",
    "BMI",
    "TOTCHOL",
    "GLUCOSE",
    "HEARTRTE",
    "CVD_MULTI"
]

corr = cvd_imputed_normalized[corr_vars].corr()

st.write(
    "The correlation matrix below illustrates associations between age, major cardiovascular risk factors, "
    "and the multiclass CVD outcome. This analysis supports the choice of age as a key variable for interactive exploration."
)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)
st.pyplot(fig)

st.caption(
    "Figure 8: Correlation matrix of selected cardiovascular risk factors and the CVD outcome."
)

st.subheader("Interpretation")
st.write(
    "The descriptive and visual analyses confirm known epidemiological patterns. "
    "Participants who developed CVD tend to be older and exhibit higher levels of established "
    "cardiovascular risk factors, such as elevated systolic blood pressure. "
    "The interactive visualisations further highlight the strong association between age and CVD risk."
)

st.title('Data analysis')

#Prepare data for modeling 
cvd_imputed_normalized = cvd_imputed_normalized.drop(labels=["TIMECVD", "CVD"], axis=1, errors='ignore')

def splitLabels(df, target):
    X = df.loc[:, df.columns != target]
    y = df[target]
    return X, y

X, y = splitLabels(cvd_imputed_normalized, target="CVD_MULTI") 

st.subheader("Feature engineering")

#Anove feature importance
from sklearn.feature_selection import SelectKBest, f_classif

feature_names = X.columns.tolist()

best_features = SelectKBest(score_func=f_classif, k="all")
fit = best_features.fit(X, y)

featureScores = pd.DataFrame(
    data=fit.scores_,
    index=feature_names,
    columns=["ANOVA Score"]
).sort_values(by="ANOVA Score", ascending=False)

fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(
    featureScores,
    annot=True,
    linewidths=0.4,
    linecolor="black",
    fmt=".2f",
    ax=ax
)
ax.set_title("Selection of Features by ANOVA score")
st.pyplot(fig)

#Interactive feature selection 
st.subheader("Interactive feature selection")

dropdown_featureselection = st.selectbox("Select", ["Threshold", "Top values"], index=1)

if dropdown_featureselection == "Threshold":
    threshold = st.slider(
        "Please select threshold value",
        0.0,
        float(featureScores["ANOVA Score"].max()),
        30.0
    )
    selected_features = featureScores[featureScores["ANOVA Score"] >= threshold].index.tolist()
else:
    top_k = st.slider(
        "Please select the amount of top features",
        1,
        len(featureScores),
        10
    )
    selected_features = featureScores.head(top_k).index.tolist()

st.write(f"Selected features ({len(selected_features)}):")
st.write(selected_features)

if len(selected_features) == 0: 
    st.warning("no features selected. Lower the threshold or choose top values.")
    st.stop()

# Create X/y based on selected features
X_selected = cvd_imputed_normalized[selected_features]
y_selected = cvd_imputed_normalized["CVD_MULTI"]


#Result table 
resultsTable = pd.DataFrame(columns=[
    "Model", "Accuracy", "F1 Score", "Precision", "Recall", "ROC_AUC", "ROC", "cm"
])

def modelResults(model, accuracy, f1, precision, recall, roc_auc, roc_cur, cm):
    model_name = str(model).split("(")[0]
    st.write(f" Model {model_name} evaluated.")
    resultsTable.loc[len(resultsTable)] = [model_name, accuracy, f1, precision, recall, roc_auc, roc_cur, cm]


#Logistic regression
def trainLogReg(X, y, max_iter=1000, test_size=0.2, class_weight='balanced'):

  # Step 1: we use the same train-test split as above
  train_X, test_X, train_y, test_y = train_test_split(
      X, y, test_size=test_size, stratify=y, random_state=380
  )

  from sklearn.preprocessing import MinMaxScaler, label_binarize
  min_max_scaler = MinMaxScaler()
  train_X = min_max_scaler.fit_transform(train_X)
  test_X = min_max_scaler.transform(test_X)

  # Step 2: Pick the algorithm
  logreg = LogisticRegression(max_iter=max_iter, class_weight=class_weight)

  # Step 3: Train the classifier
  logreg = logreg.fit(train_X, train_y)

  # Step 4: Make a prediction
  y_pred = logreg.predict(test_X)
  y_pred_proba = logreg.predict_proba(test_X)

  # Cross validation
  cv_value = 5
  cv_scores = cross_val_score(logreg, train_X, train_y, scoring='accuracy', cv=cv_value)

  # Step 5: Evaluate the prediction
  st.write('Used model is:', logreg)
  st.text(classification_report(test_y, y_pred))
  st.write(
      f"Cross validation score: {cv_scores.mean():.2f} "
      f"± {cv_scores.std():.2f}"
  )

  cm = confusion_matrix(test_y, y_pred, normalize='true')

  fig, ax = plt.subplots()
  disp = ConfusionMatrixDisplay(
      confusion_matrix=cm,
      display_labels=logreg.classes_
  )
  disp.plot(ax=ax)
  st.pyplot(fig)

  # Metrics
  average = 'macro'
  multi_class = 'ovo'

  accuracy = accuracy_score(test_y, y_pred)
  f1 = f1_score(test_y, y_pred, average=average)
  precision = precision_score(test_y, y_pred, average=average, zero_division=0)
  recall = recall_score(test_y, y_pred, average=average, zero_division=0)
  roc_auc = roc_auc_score(
      test_y, y_pred_proba,
      average=average,
      multi_class=multi_class
  )

  y_test_binary = label_binarize(test_y, classes=logreg.classes_)
  roc_cur = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())

  modelResults(logreg, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)

#KNN
def trainKNN(X, y, neighbors=5, test_size=0.2):
  # Step 1: we use the same train-test split as above
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, stratify=y, random_state=380)

  from sklearn.preprocessing import label_binarize

  # Step 2: Pick the algorithm
  knn = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform')

  # Step 3: Train the classifier
  knn = knn.fit(train_X, train_y)

  # Step 4: Make a prediction
  y_pred = knn.predict(test_X)
  y_pred_proba = knn.predict_proba(test_X) #For ROC curve specifically

  #Cross validation:
  cv_value = 5
  cv_scores = cross_val_score(knn, train_X, train_y, scoring='accuracy', cv=cv_value)

  # Step 4: Make a prediction
  prediction = cross_val_predict(knn, test_X, test_y, cv=cv_value)

  # Step 5: Evaluate the prediction
  st.write('Used model is: {}-----------'.format(knn))
  st.write('classifier stored as "knn"')
  st.text(classification_report(test_y, y_pred, zero_division=0))
  st.write(f"Cross validation score: {cv_scores.mean().round(2)} accuracy with a standard deviation of {cv_scores.std().round(2)}")

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  fig, ax = plt.subplots()
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
  disp.plot(ax=ax)
  st.pyplot(fig)

  #Make extra section for table that contains all important information for AUC curve
  average='macro'
  multi_class='ovo'

  accuracy = accuracy_score(test_y, y_pred)
  f1 = f1_score(test_y, y_pred, average=average)
  precision = precision_score(test_y, y_pred, average=average, zero_division=0)
  recall = recall_score(test_y, y_pred, average=average, zero_division=0)
  roc_auc = roc_auc_score(test_y, y_pred_proba, average=average, multi_class=multi_class)

  #Binarise output in one vs all fashion (one group compared to two others)
  y_test_binary = label_binarize(test_y, classes=knn.classes_)

  roc_cur = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  modelResults(knn, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)

#Random forest 
def trainRF(X, y, test_size = 0.2, max_depth = 4, min_samples_leaf = 1):

  # Step 1: we use the same train-test split as above
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = test_size, stratify = y, random_state = 380)

  from sklearn.preprocessing import label_binarize

  # Step 2: Pick the algorithm
  RF = RandomForestClassifier(max_depth = max_depth, min_samples_leaf = min_samples_leaf, random_state=380, class_weight='balanced')

  # Step 3: Train the classifier
  RF.fit(train_X, train_y)

  # Step 4: Make a prediction
  y_pred = RF.predict(test_X)
  y_pred_proba = RF.predict_proba(test_X)

  #Cross validation:
  cv_value = 5
  cv_scores = cross_val_score(RF, train_X, train_y, scoring='accuracy', cv=cv_value)

  # Step 4: Make a prediction
  prediction = cross_val_predict(RF, test_X, test_y, cv=cv_value)

  # Step 5: Evaluate the prediction
  st.write('Used model is: {}-----------'.format(RF))
  st.write('classifier stored as "RF"')
  st.text(classification_report(test_y, y_pred, zero_division=0))
  st.write(f"Cross validation score: {cv_scores.mean().round(2)} accuracy with a standard deviation of {cv_scores.std().round(2)}")

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  fig, ax = plt.subplots()
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RF.classes_)
  disp.plot(ax=ax)
  st.pyplot(fig)

  #Make extra section for table that contains all important information for AUC curve
  average='macro'
  multi_class='ovo'

  accuracy = accuracy_score(test_y, y_pred)
  f1 = f1_score(test_y, y_pred, average=average)
  precision = precision_score(test_y, y_pred, average=average, zero_division=0)
  recall = recall_score(test_y, y_pred, average=average, zero_division=0)
  roc_auc = roc_auc_score(test_y, y_pred_proba, average=average, multi_class=multi_class)

  y_test_binary = label_binarize(test_y, classes=RF.classes_)
  roc_cur = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  modelResults(RF, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)

st.subheader("Comparison of predictive models")

# Interactive modelling with outcomes
col1, col2, col3 = st.columns(3)

with col1:
    run_lr = st.button("Run Logistic Regression")
with col2:
    run_knn = st.button("Run KNN")
with col3:
    run_rf = st.button("Run Random Forest")

if run_lr:
    trainLogReg(X_selected, y_selected)

if run_knn:
    trainKNN(X_selected, y_selected)

if run_rf:
    trainRF(X_selected, y_selected)

# Show results table if anything has been run
if len(resultsTable) > 0:
    st.subheader("Results table")
    st.dataframe(
        resultsTable.drop(columns=["ROC", "cm"]),
        use_container_width=True
    )

    # Best model by F1
    best_model = resultsTable.loc[
        resultsTable['F1 Score'].idxmax()
    ]
    st.write(
        f"The model with the highest F1 score was "
        f"**{best_model['Model']}** "
        f"with an F1 score of **{best_model['F1 Score']:.4f}**"
    )

    # ROC curves comparison
    st.subheader("ROC Curves - Model Comparison")

    fig, ax = plt.subplots(figsize=(8, 6))

    for _, row in resultsTable.iterrows():
        model_name = row['Model']
        fpr, tpr, _ = row['ROC']
        roc_auc = row['ROC_AUC']
        ax.plot(
            fpr, tpr, lw=2,
            label=f'{model_name} (AUC = {roc_auc:.3f})'
        )

    ax.plot(
        [0, 1], [0, 1],
        'k--', lw=2,
        label='Random Classifier (AUC = 0.500)'
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend(loc="lower right")
    ax.grid(True)

    st.pyplot(fig)

st.subheader('Hyperparameter Optimisation')
st.write('To further enhance model performance, we iterate through a range of hyperparaneters' \
 ' for KNN and Random Forest models. (logistic regression does not have signiificant hyperparameters to tune)'
 ' The optimal hyperparameters are selected based on F1 score. The F1 score was specifically selected'
 ' as risk assessment for CVD development (i.e. to allocate preventive measures) requires both good recall'
 ' and precision, as not to overload healthcare providers with much burden while also detecting as many cases'
 ' as possible.')

st.write('The following range of hyperparameters were tested:')
"""
- KNN: number of neighbors from 1 to 20
- Random Forest: max depth from 2 to 7, min samples leaf as 1 or 10 (mainly to avoid overfitting)
"""

### REDEFINING SOME TRAINING FUNCTIONS FOR HYPERPARAMETER OPTIMISATION
##Removing the text part so it doesn't flood the screen

def modelResults_hyperparameters(model, accuracy, f1, precision, recall, roc_auc, roc_cur, cm):
    resultsTable.loc[len(resultsTable)] = [model, accuracy, f1, precision, recall, roc_auc, roc_cur, cm]

def trainRF_hyperparameters(X, y, test_size=0.2, max_depth=4, min_samples_leaf=1):

  # Step 1: we use the same train-test split as above
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, stratify=y, random_state=380)

  from sklearn.preprocessing import label_binarize

  # Step 2: Pick the algorithm
  RF = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=380, class_weight='balanced')

  # Step 3: Train the classifier
  RF.fit(train_X, train_y)

  # Step 4: Make a prediction
  y_pred = RF.predict(test_X)
  y_pred_proba = RF.predict_proba(test_X) #For ROC curve specifically

  #Cross validation:
  cv_value = 5
  cv_scores = cross_val_score(RF, train_X, train_y, scoring='accuracy', cv=cv_value)

  # Step 4: Make a prediction
  prediction = cross_val_predict(RF, test_X, test_y, cv=cv_value)


  cm = confusion_matrix(y_true=test_y,
                  y_pred=y_pred,
                  normalize='true')

  #Make extra section for table that contains all important information for AUC curve
  average='macro' #setting since we do not have binary classification problem anymore
                  #important to use macro since this does not consider class imbalance (basically weighs classes all the same)
  multi_class='ovo' #one vs one, makes it insensitive to class imbalance if average=macro

  accuracy = accuracy_score(test_y, y_pred)
  f1 = f1_score(test_y, y_pred, average=average)
  precision = precision_score(test_y, y_pred, average=average)
  recall = recall_score(test_y, y_pred, average=average)
  roc_auc = roc_auc_score(test_y, y_pred_proba, average=average, multi_class=multi_class)

  #To get ROC curve we need fpr, tpr, threshholds, this can only be done with binary
  #Binarise output in one vs all fashion (one group compared to two others)
  y_test_binary = label_binarize(test_y, classes=RF.classes_)

  #This should be micro-averaged, so keep in mind class imbalance has effect
  roc_cur = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  modelResults_hyperparameters(RF, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)#add roc_cur after roc_auc for binary problem

def trainKNN_hyperparameters(X, y, neighbors = 5, test_size=0.2):
  # Step 1: we use the same train-test split as above
  train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size, stratify=y, random_state=380)

  from sklearn.preprocessing import label_binarize

  # Step 2: Pick the algorithm
  # TODO: HOW DO WE DEFINE THE CLASSIFIER?
  ### YOUR CODE HERE
  knn = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform')

  # Step 3: Train the classifier
  knn = knn.fit(train_X, train_y)

  # Step 4: Make a prediction
  y_pred = knn.predict(test_X)
  y_pred_proba = knn.predict_proba(test_X) #For ROC curve specifically

  #Cross validation:
  cv_value = 5
  cv_scores = cross_val_score(knn, train_X, train_y, scoring='accuracy', cv=cv_value)

  # Step 4: Make a prediction
  prediction = cross_val_predict(knn, test_X, test_y, cv=cv_value)

  cm = confusion_matrix(y_true=test_y,
                  y_pred=y_pred,
                  normalize='true')


  #Make extra section for table that contains all important information for AUC curve
  average='macro' #setting since we do not have binary classification problem anymore
                  #important to use macro since this does not consider class imbalance (basically weighs classes all the same)
  multi_class='ovo' #one vs one, makes it insensitive to class imbalance if average=macro

  accuracy = accuracy_score(test_y, y_pred)
  f1 = f1_score(test_y, y_pred, average=average)
  precision = precision_score(test_y, y_pred, average=average)
  recall = recall_score(test_y, y_pred, average=average)
  roc_auc = roc_auc_score(test_y, y_pred_proba, average=average, multi_class=multi_class)

  #To get ROC curve we need fpr, tpr, threshholds, this can only be done with binary
  #Binarise output in one vs all fashion (one group compared to two others)
  y_test_binary = label_binarize(test_y, classes=knn.classes_)

  #This should be micro-averaged, so keep in mind class imbalance has effect
  roc_cur = roc_curve(y_test_binary.ravel(), y_pred_proba.ravel())

  cm = confusion_matrix(y_true=test_y, y_pred=y_pred, normalize='true')

  modelResults_hyperparameters(knn, accuracy, f1, precision, recall, roc_auc, roc_cur, cm)#add roc_cur after roc_auc for binary problem


### FUNCTIONS FOR OUR ITERATIONS BETWEEN HYPERPARAMETERS
def hyperparameter_KNN(max_neighbours):
  for i in range (4, max_neighbours + 1):
    trainKNN_hyperparameters(X, y, neighbors = i, test_size=0.2)

def hyperparameter_RF(max_depth, min_samples_leaf):
  for i in range(2, max_depth):
    trainRF_hyperparameters(X, y, max_depth=i, min_samples_leaf=min_samples_leaf)

### HYPERPARAMETER ITERATIONS
hyperparameter_KNN(10)
hyperparameter_RF(5, 10) #second argument is the min samples leaf
hyperparameter_RF(5, 1)

## removing duplicate rows if they were created by accident
#removing accidental duplicate models
resultsTable = resultsTable.drop_duplicates(subset=['Model']) #, 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC_AUC'

#Top model
best_model = resultsTable.loc[resultsTable['F1 Score'].idxmax()]
st.write(f"The model with the highest F1 score was {best_model['Model']} with an F1 score of {best_model['F1 Score']:.4f}")

st.write('Below is a table summarizing the top 3 performing models (ranked by F1 score), including logistic regression as a benchmark.')

#Table
top3_models = resultsTable.sort_values(by='F1 Score', ascending=False).head(3)
top3_models = pd.concat([top3_models, resultsTable.head(1)])
top3_models['Model'] = top3_models['Model'].astype(str)
st.dataframe(top3_models.drop(labels='cm', axis=1)) #For some reason, only in streamlit the cm column caused issues

fig, ax = plt.subplots(figsize=(10, 10)) #Redo this so it uses correct style, took this from previous project and only made it work so far

for idx, row in top3_models.iterrows():
    model_name = str(row['Model']).split('(')[0]
    fpr, tpr, thresholds = row['ROC']
    roc_auc = row['ROC_AUC']
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')

st.write('The same models are compared below using ROC curves:')

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.0])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14)
ax.legend(loc="lower right", fontsize=15)
fig.tight_layout()
ax.grid()

st.pyplot(fig)

st.title('References')
st.write(' 1. World Health Organisation (2025), "Cardiovascular diseases (CVDs)", https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)')


