# Data structures
import pandas as pd
import numpy as np

# Viz
import plotly.express as px
import streamlit as st


### SCIKIT-LEARN ###
from sklearn.pipeline import make_pipeline

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
import pickle


#####################################################################################################################################################
#                                                                                                                                                   #
#                                                           PREDICTION FUNCTION                                                                     #
#                                                                                                                                                   #
#####################################################################################################################################################

def encode_categorical(df):
    '''Translate categorical columns to numeric (ordinal or onehot encoding)'''
    map_Gender = {'Male' : -1, 'Female' : 1}
    map_Married = {'Yes' : -1, 'No' : 1}
    map_Education = {'Graduate' : -1, 'Not Graduate' : 1}
    map_Dependents = {'0' : 0,  '1' : 1,  '2' : 2,  '3+' : 3}
    map_Self_Employed = {'Yes' : -1,  'No' : 1}
    map_Property_Area = {'Urban' : 1,  'Semiurban' : 2, 'Rural' : 3}
    map_Loan_Amount_Term = {360.0 : 1, 180.0 : 0, 480.0 : 0,
                            300.0 : 0, 240.0 : 0, 84.0 : 0,
                            6.0 : 0, 120.0 : 0, 36.0 : 0,
                            36.0  : 0, 350.0 : 0, 12.0 : 0}

    
    # ordinal encoding
    df['Gender'] = df['Gender'].map(map_Gender)
    df['Married'] = df['Married'].map(map_Married)
    df['Education'] = df['Education'].map(map_Education)
    df['Dependents'] = df['Dependents'].map(map_Dependents)
    df['Self_Employed'] = df['Self_Employed'].map(map_Self_Employed)
    df['Property_Area'] = df['Property_Area'].map(map_Property_Area)
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].map(map_Loan_Amount_Term)

    return df


def featureEngineering(df):
    df = df.drop(['Loan_ID'], axis=1)
    df['CoapplicantIncome'] = df['CoapplicantIncome'].replace(0, np.nan, inplace=True)
    df = df.drop(df[df['CoapplicantIncome'] > 30000].index)

    df['ApplicantIncome'] = np.where((df.ApplicantIncome > 30000),30000,df.ApplicantIncome)

    return df
    


def processData(df):
    """Process data"""
    df = encode_categorical(df)
    df = featureEngineering(df)


    if 'Loan_Status' in df.columns:
        df['Loan_Status'] = df['Loan_Status'].map({'Y' : 1,
                                                   'N' : 0})
    return df


def create_model():
    ### GET DATA
    print('get data')
    df_train = pd.read_csv('data/train.csv')
    df_train = processData(df_train)

    imputer = IterativeImputer(random_state=0)
    imputer.fit_transform(df_train,y=None)

    y_train = df_train['Loan_Status']
    x_train =  df_train.drop(['Loan_Status'], axis=1)
    print(x_train.values[0])


    # Model creation
    preprocessor = make_pipeline(IterativeImputer(random_state=0),SelectKBest(k=8))
    Logistic = LogisticRegression(random_state=0)

    model = make_pipeline(preprocessor, Logistic)
    model.fit(x_train,y_train)

    pickle.dump(model, open('model.sav', 'wb'))

    return pd.read_csv('data/train.csv'), model


def load_models():
    return pickle.load(open('model.sav', 'rb'))


def predict(X, model):
    return model.predict_proba(np.array([X]))[0]
        










#####################################################################################################################################################
#                                                                                                                                                   #
#                                                           DASHBOARD WITH STREAMLIT                                                                #
#                                                                                                                                                   #
#####################################################################################################################################################
#
df, _ = create_model()
# df = pd.read_csv('data/train.csv')
model = load_models()
x = [-1.0, 1.0, 0.0, -1, 1.0, 5849, None, None, 1.0, 1.0, 1]
##print(df)
##print(predict(x, model))

##input()

st.title('Loan prediction')

###############
st.write("Automation of the loan eligibility process")
st.write("Would your loan be approved ?")
     

### Sidebar
df = df.dropna()

st.sidebar.title('Parameters')

Gender = st.sidebar.radio(
    'Gender',
     ['Non-binary']+list(df['Gender'].unique()))
Married = st.sidebar.radio(
    'Married ?',
     list(df['Married'].unique()))
Education = st.sidebar.radio(
    'Education',
     list(df['Education'].unique()))
Self_Employed = st.sidebar.radio(
    'Self employed ?',
     list(df['Self_Employed'].unique()))
ApplicantIncome = st.sidebar.select_slider(
    'Applicant income',
     [i for i in range(10001)])
LoanAmount = st.sidebar.select_slider(
    'Loan amount (in thousands)',
     [i for i in range(301)])
Property_Area = st.sidebar.radio(
    'Habitation area',
    list(df['Property_Area'].unique()))

l = [Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Property_Area]
for i in range(len(l)):
    if l[i] == 'Unknown' or l[i] == 'Non-binary':
        l[i] = None
[Gender, Married, Education, Self_Employed, ApplicantIncome, LoanAmount, Property_Area] = l

df_x = pd.DataFrame({
    "Loan_ID" : [None],
    "Gender" : [Gender],
    "Married" : [Married],
    "Dependents" : [None],
    "Education" : [Education],
    "Self_Employed" : [Self_Employed],
    "ApplicantIncome" : [ApplicantIncome],
    "CoapplicantIncome" : [None],
    "LoanAmount" : [LoanAmount],
    "Loan_Amount_Term" : [360],
    "Credit_History" : [None],
    "Property_Area" : [Property_Area]})


### Prediction
df_x = processData(df_x)

pred = model.predict_proba(df_x)[0]


fig3 = px.pie(["Not accepted", "Accepted"], values=[pred[0], pred[1]],color=["Not accepted", "Accepted"],
color_discrete_map={'Not accepted':'red', 'Accepted':'green'})
fig3.update_layout(
title="<b>Loan approved ?</b>")
st.plotly_chart(fig3)




expander = st.expander("More about this project")
expander.write("You can email me: hkfuoc98@gmail.com")