import pickle

import os

import streamlit as st

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import sklearn
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

from PIL import Image


# load model
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'best_model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# load test data
TEST_DATA_PATH = os.path.join(os.getcwd(), 'data', 'test_data.xlsx')
df_test = pd.read_excel(TEST_DATA_PATH, engine='openpyxl')
y_test = df_test['output']
X_test = df_test.drop('output', axis=1)

st.title('Heart Attack Predictor')

# st.header('Background')

# st.markdown('''As of 2020, according to the data published by the World Health Organization (WHO), 
# coronary heart disease (an example of cardiovascular disease) is the #1 leading cause of death in Malaysia. 
# The number of death caused by it reached 36,729 deaths amounting to 21.86% of the total deaths in Malaysia. 
# This puts Malaysia as the rank #61 country with the highest age-adjusted death rate (136 per 100,000 population).
# [Source](https://www.worldlifeexpectancy.com/malaysia-coronary-heart-disease
# )''')

st.header("Heart Attack Predictor Tool")

with st.form("heart attack form"):
    st.write("Insert your details")

    age = int(st.slider('Select your age', 
                         min_value=0,
                         max_value=100, 
                         value=50, 
                         step=1))
    gender = st.selectbox('Select your gender',
                           options=['Male', 'Female'],
                           index=0)
    chest_pain = st.selectbox('Chest pain type',
                               options=['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
                               index=3)
    rest_bp = float(st.number_input('Resting blood pressure (mm Hg)', 
                                     value=120))
    chol = float(st.number_input('Cholesterol level (mg/dL)', 
                                  value=240))
    fast_blood_sugar = st.selectbox('Is your fasting blood sugar more than 120 mg/dL?',
                                     options=['Yes', 'No'],
                                     index=1)
    rest_ecg = st.selectbox('Resting electrocardiographic results',
                             options=['Normal', 'Abnormal', 'Critical'],
                             index=0)
    max_heart_rate = float(st.number_input('Max heart rate', 
                                            value=150))
    excs_ind_angina = st.selectbox('Do you have exercise-induced angina?',
                                    options=['Yes', 'No'],
                                    index=1)
    old_peak = float(st.number_input('ST depression induced by exercise relative to rest', 
                                      value=1.0))
    slp = st.selectbox('Slope of the peak exercise ST segment',
                            options=['Downsloping', 'Flat', 'Upsloping'],
                            index=2)
    caa = int(st.selectbox('Number of major vessels colored by flourosopy',
                            options=['0', '1', '2', '3', '4'],
                            index=0))
    thall = st.selectbox('Blood defect',
                              options=['Normal', 'Reversible defect', 'Fixed defect'],
                              index=2)

    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")

    if submitted:
        st.write('Details given are:')
        st.write('Age', age, 
                 'Gender', gender, 
                 'Chest pain type', chest_pain,
                 'Resting blood pressure', rest_bp,
                 'Cholesterol level', chol,
                 'Fasting blood sugar', fast_blood_sugar,
                 'Resting ECG result', rest_ecg,
                 'Max heart rate', max_heart_rate,
                 'Excercise-induced angina', excs_ind_angina,
                 'Oldpeak', old_peak,
                 'SLP', slp,
                 'CAA', caa,
                 'THALL', thall)

        gender_mapped=1 if gender=='Male' else 0

        if chest_pain == 'Typical angina':
            chest_pain_mapped = 0
        elif chest_pain == 'Atypical angina':
            chest_pain_mapped = 1
        elif chest_pain == 'Non-anginal pain':
            chest_pain_mapped = 2
        elif chest_pain == 'Asymptomatic':
            chest_pain_mapped = 3
        else:
            chest_pain_mapped = 0

        fast_blood_sugar_mapped=1 if fast_blood_sugar=='Yes' else 0

        if rest_ecg == 'Normal':
            rest_ecg_mapped = 0
        elif rest_ecg == 'Abnormal':
            rest_ecg_mapped = 1
        elif rest_ecg == 'Critical':
            rest_ecg_mapped = 2
        else:
            rest_ecg_mapped = 0

        excs_ind_angina_mapped=1 if excs_ind_angina=='Yes' else 0

        if slp == 'Downsloping':
            slp_mapped = 0
        elif slp == 'Flat':
            slp_mapped = 1
        elif slp == 'Upsloping':
            slp_mapped = 2
        else:
            slp_mapped = 0

        if thall == 'Normal':
            thall_mapped = 2
        elif thall == 'Reversible defect':
            thall_mapped = 3
        elif thall == 'Fixed defect':
            thall_mapped = 1
        else:
            thall_mapped = 0

        input_row = np.array([age, 
                              gender_mapped, 
                              chest_pain_mapped, 
                              rest_bp, 
                              chol, 
                              fast_blood_sugar_mapped,
                              rest_ecg_mapped, 
                              max_heart_rate, 
                              excs_ind_angina_mapped, 
                              old_peak, 
                              slp_mapped,
                              caa, 
                              thall_mapped])

        input_data = input_row.reshape(1, -1)
        y_pred = model.predict(input_data)[0]
        y_list = ['lower chance of getting heart attack', 'higher chance of getting heart attack']
        y_result = y_list[y_pred]

        probas = model.predict_proba(input_data)[0]
        proba = probas[y_pred]
    
        st.write(f'You have a {y_result}. Probability of approximately {proba*100: .2f}\%')

st.header("Test Case")

st.write('''Below is an example of a test dataset that will be used by the trained model for generating predictions for each sample 
and will be compared with the true label.''')

st.write(df_test)

score = model.score(X_test, y_test)

st.markdown(f'''Based on our trained model, testing using the above dataset, we achieved an accuracy score of {score}. We show additional
informations below i.e. the classification report and the confusion matrix.''')

y_pred = model.predict(X_test)

IMAGE_PATH = os.path.join(os.getcwd(), 'statics', 'classification_report.png')
image = Image.open(IMAGE_PATH)
st.image(image, caption='Test case classification report')

cm = confusion_matrix(y_test, y_pred, normalize='all')
cmd = ConfusionMatrixDisplay(cm)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
cmd.from_predictions(y_test, 
                     y_pred, 
                     normalize='all', 
                     ax=ax, 
                     display_labels=['lower heart attack chance', 'higher heart attack chance'],
                     xticks_rotation=10)
st.write(fig)