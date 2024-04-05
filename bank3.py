import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
#from sklearn.ensemble import DecisionTreeClassifier # type: ignore
#from sklearn.preprocessing import LabelEncoder # type: ignore
#from sklearn.tree import DecisionTreeClassifier # type: ignore
import string
import requests # type: ignore

# Load the pre-trained model
#@st.cache
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()



def lg0(LoanAmount):
    return np.log(float(LoanAmount)+1) 

def lg1(ApplicantIncome):
    return np.log(float(ApplicantIncome)+1) 

def lg2(Loan_Amount_Term):
    return np.log(float(Loan_Amount_Term)+1) 

def lg3(CoapplicantIncome):
    return np.log(float(CoapplicantIncome)+1)  


def lg4(Total_Income):
    return np.log(float(Total_Income)+1) 


# Define function to calculate total income
def calculate_total_income(applicant_income, coapplicant_income):
    return applicant_income + coapplicant_income



# Define preprocessing function
def preprocess_data(Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, LoanAmount, ApplicantIncome, Loan_Amount_Term, CoapplicantIncome):
    # Perform any necessary preprocessing
    # Convert categorical variables to numerical values
    gender_map = {'Male': 0, 'Female': 1}
    married_map = {'No': 0, 'Yes': 1}
    education_map = {'Graduate': 0, 'Not Graduate': 1}
    employed_map = {'No': 0, 'Yes': 1}
    property_area_map = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

    Gender = gender_map.get(Gender, -1)
    Married = married_map.get(Married, -1)
    Education = education_map.get(Education, -1)
    Self_Employed = employed_map.get(Self_Employed, -1)
    Property_Area = property_area_map.get(Property_Area, -1)

    # Calculate total income
    Total_Income = calculate_total_income(ApplicantIncome, CoapplicantIncome)
    LoanAmount = lg0(LoanAmount)
    ApplicantIncome= lg1(ApplicantIncome)
    Loan_Amount_Term= lg2(Loan_Amount_Term)
    CoapplicantIncome= lg3(CoapplicantIncome)
    Total_Income= lg4(Total_Income)

    # Return the preprocessed data as a list
    return [Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, LoanAmount, ApplicantIncome, Loan_Amount_Term, CoapplicantIncome,Total_Income]

# Define function to make predictions
def predict_loan_approval(data):
    prediction = model.predict([data])
    return prediction[0]

def main():
    st.title("Bank Loan Authenticator")
    Gender = st.selectbox("Gender", ['Male', 'Female'])
    Married = st.selectbox("Married", ['No', 'Yes'])
    Dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)
    Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self Employed", ['No', 'Yes'])
    Credit_History = st.number_input("Credit History", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    Property_Area = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])
    LoanAmount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=0)
    ApplicantIncome = st.number_input("Applicant Income")
    Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0, max_value=500, value=0)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, max_value=100000, value=0)
    

    # Preprocess the input data
    data = preprocess_data(Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, LoanAmount, ApplicantIncome, Loan_Amount_Term, CoapplicantIncome)

    # Make prediction
    if st.button("Predict"):
        prediction = predict_loan_approval(data)
        if prediction == 1:
            st.success("Congratulations! Your loan is approved.")
        else:
            st.error("Sorry, your loan application is rejected.")

if __name__ == '__main__':
    main()
