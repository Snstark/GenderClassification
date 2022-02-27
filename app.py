import streamlit as st
import pandas as pd

import pickle
import numpy as np

st.title("Gender Classification")
st.write("Gender classification Using Facial features")

data = pd.read_csv('data//gender_classification_v7.csv')
DT = pickle.load(open('GenderDT.pkl', 'rb'))
LogReg = pickle.load(open('GenderLogReg.pkl', 'rb'))
RFC = pickle.load(open('GenderRFC.pkl', 'rb'))

nav = st.sidebar.radio('Navigation', ['Home', 'Prediction'])
if nav == 'Home':
    st.title('ABC Corporation')
    st.subheader('Gender Classification Using Facial Features')
    if st.checkbox('Show Data'):
        st.dataframe(data)
if nav == 'Prediction':
    st.subheader('Please give the following information:')

    long_hair = st.number_input('Does the person have long Hair  if YES then give 1  NO then give 0')

    forehead_width_cm = st.number_input("Enter the Forehead Width in Cm")

    forehead_height_cm = st.number_input("Enter Forehead height in CM", max_value=7.2)

    nose_wide = st.number_input("If Nose is wide then give 1 if NOT then give 0 ")

    nose_long = st.number_input("If Nose is Long then give 1 if NOT then 0")

    lips_thin = st.number_input("If Lips are thin then give 1 if NOT then give 0 ")

    distance_nose_to_lip_long = st.number_input("If distance from Nose to Lip is long then give 1 if NOT 0 ")

    x = np.array([long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin,
                  distance_nose_to_lip_long])

    x = x.reshape(1, 7)

    st.header("Select the Classifier")
    Cls = st.sidebar.radio('Algorithm', ['DecisionTreeClassifier', 'LogisticsRegression',
                                         'RandomForestClassifier', "ALL"])
    if Cls == "DecisionTreeClassifier":
        if st.button('Predict'):
            S = DT.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

    if Cls == "LogisticsRegression":
        if st.button('Predict'):
            S = LogReg.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")
    if Cls == "RandomForestClassifier":
        if st.button('Predict'):
            S = RFC.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")
    if Cls == "ALL":
        if st.button('Predict'):
            st.header("RandomForestClassifiers")
            A = RFC.predict(x)
            if A == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

            st.header("LogisticRegression")
            S = LogReg.predict(x)
            if S == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")

            st.header("DecisionTreeClassifier")
            C = DT.predict(x)
            if C == 0:
                st.header("Based on features gender is Female")
            else:
                st.header("Based on features gender is male")
