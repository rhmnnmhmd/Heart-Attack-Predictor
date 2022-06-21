# Libraries
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns

import pickle

import scipy as sp
from scipy import stats

import sklearn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

# # User-defined functions
def categorical_matrix_display(df, columns):
    dim = len(columns)
    array = np.zeros((dim, dim))          

    for i, name1 in enumerate(columns):
        for j, name2 in enumerate(columns):
            logit = LogisticRegression()
            logit.fit(df[name1].values.reshape(-1, 1), df[name2])
            score = logit.score(df[name1].values.reshape(-1, 1), df[name2])
            array[i, j] = score

    arrayFrame = pd.DataFrame(data=array, columns=columns)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(arrayFrame, annot=True, ax=ax, yticklabels=columns, vmin=0, vmax=1)

def cramers_V(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def cramersVMatrix(df, col):
    len_cat = len(col)
    array  = np.zeros((len_cat, len_cat))

    for i, name1 in enumerate(col):
        for j, name2 in enumerate(col):
            cross_tab = pd.crosstab(df[name1], df[name2]).to_numpy()
            value = cramers_V(cross_tab)
            array[i, j] = value

    array_frame = pd.DataFrame(data=array, columns=col)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(array_frame, annot=True, ax=ax, yticklabels=col, vmin=0, vmax=1)

# # Data loading
DATA_PATH = os.path.join(os.getcwd(), 'data', 'heart.csv')
df = pd.read_csv(DATA_PATH)

# ## rename columns to be more descriptive
my_dict = {'cp': 'chest_pain', 
           'trtbps': 'rest_bp', 
           'fbs': 'fast_blood_sugar', 
           'restecg': 'rest_ecg',
           'thalachh': 'max_heart_rate',
           'exng': 'excs_ind_angina',}
df = df.rename(columns=my_dict)

# # General infos
df.info()

# # Basic descriptie statistics
df.describe()

# # Check NAs
df.isna().sum()

# # Check duplicates
df.duplicated().sum()

# show the duplicated rows
df[df.duplicated()]

# ## drop duplicates
df = df.drop_duplicates()

# # EDA/Data wrangling
# ## handling 'thall' column
df['thall'].value_counts()

# drop the two rows with 'thall' = 0
df = df.loc[df['thall']!=0]

# ## numerical features
df_num = df[['age', 'rest_bp', 'chol', 'max_heart_rate', 'oldpeak']]

# ### boxplots for each numerical features to find outliers
fig, ax = plt.subplots(5, 1, figsize=(6, 4))
df_num.plot.box(layout=(5, 1), 
            subplots=True, 
            ax=ax, 
            vert=False, 
            sharex=False)
plt.tight_layout()
plt.show()

# ### remove outliers (the dots/circles in the above boxplots)
columns = df_num.columns[1: ]

for i, col in enumerate(columns):
    q1 = np.quantile(df[col], 0.25)
    q3 = np.quantile(df[col], 0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr
    df = df.loc[(df[col]>lower) & (df[col]<upper)]

# ### the dataset's basic statistics after outlier removal
df.describe()

# ### correlation matrix
fig, ax = plt.subplots(1, 1, figsize = (5, 5))
sns.heatmap(df_num.corr(), annot = True, ax=ax)
plt.tight_layout()
plt.show()

# ## categorical features
df_cat = df.drop(['age', 'rest_bp', 'chol', 'max_heart_rate', 'oldpeak'], axis=1)

# ### countplots for each categorical features
cat_cols = df_cat.columns

for i, col in enumerate(cat_cols):
    ax = sns.countplot(x=df[col], order=df[col].value_counts(ascending=False).index)
    abs_values = df[col].value_counts(ascending=False)
    rel_values = df[col].value_counts(ascending=False, normalize=True).values*100
    lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]
    ax.bar_label(container=ax.containers[0], labels=lbls)
    plt.tight_layout()
    plt.show()  

# ### correlation matrix for categorical features using logistic regression
categorical_matrix_display(df_cat, cat_cols)

# ### correlation matrix for categorical features using cramer's V values
cramersVMatrix(df_cat, cat_cols)

# # Separate features and target
# ## features
X = df.drop('output', axis=1)

# ## target
y = df['output']

# # Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Machine learning
# ## test which pipeline is better. different combination of estimator and scaler (min-max and standard scaler)
steps_logreg_mms = [('mms', MinMaxScaler()), 
                    ('logreg', LogisticRegression())]
steps_logreg_ss = [('ss', StandardScaler()), 
                   ('logreg', LogisticRegression())]

steps_dt_mms = [('mms', MinMaxScaler()), 
                ('dt', DecisionTreeClassifier())]
steps_dt_ss = [('ss', StandardScaler()), 
               ('dt', DecisionTreeClassifier())]

steps_rf_mms = [('mms', MinMaxScaler()), 
                ('rf', RandomForestClassifier())]
steps_rf_ss = [('ss', StandardScaler()), 
               ('rf', RandomForestClassifier())]

steps_knn_mms = [('mms', MinMaxScaler()), 
                ('knn', KNeighborsClassifier())]
steps_knn_ss = [('ss', StandardScaler()), 
                ('knn', KNeighborsClassifier())]
 
steps_svc_mms = [('mms', MinMaxScaler()), 
                 ('svc', LinearSVC())]
steps_svc_ss = [('ss', StandardScaler()), 
                ('svc', LinearSVC())]

best_score = 0.0

steps = [steps_logreg_mms, 
         steps_logreg_ss, 
         steps_dt_mms, 
         steps_dt_ss,
         steps_rf_mms, 
         steps_rf_ss,
         steps_knn_mms,
         steps_knn_ss,
         steps_svc_mms,
         steps_svc_ss]

for i, step in enumerate(steps):
    pipe = Pipeline(steps=step)
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)
    print(f'Pipe {i+1} accuracy score: {score}')

    if score > best_score:
        best_score = score
        best_pipe = pipe

print(f'Best pipe is {best_pipe} with accuracy score of {best_score}')

# ### so, best pipeline is Random Forest Classifier with accuracy score of 0.8571 
# (both Min-Max and Standard scaler gave the same accuracy)

# ## GridSearchCV using Random Forest Classifier as the estimator and min-max scaler
# ### create the gridsearch object
params = {'rf__n_estimators': [25, 50, 100, 200, 400],
          'rf__criterion': ['gini', 'entropy', 'log_loss'],
          'rf__bootstrap': [True, False],
          'rf__oob_score': [False, True],
          'rf__random_state': [None, 42]}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

grid_svc = GridSearchCV(estimator=best_pipe, param_grid=params, cv=kf, verbose=1)

# ### fitting
grid_svc.fit(X_train, y_train)

# ### best estimator
grid_svc.best_estimator_

# ### best accuracy score
grid_svc.best_score_

# ### best parameters
grid_svc.best_params_

# ### save the model (fitted GridSearchCV object)
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best_model.pkl')
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(grid_svc, file)

# # Metrics
# ## classification report
y_pred = grid_svc.predict(X_test)
print(classification_report(y_true=y_test, y_pred=y_pred, target_names=['lower heart attack chance', 'higher heart attack chance']))

# ## confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='all')
cmd = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
cmd.from_predictions(y_test, 
                     y_pred, 
                     normalize='all', 
                     ax=ax, 
                     display_labels=['lower heart attack chance', 'higher heart attack chance'],
                     xticks_rotation=10)
plt.show()