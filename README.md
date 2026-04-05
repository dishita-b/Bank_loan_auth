
# 🏦 Bank Loan Approval Predictor

Predicts loan approval using a trained Decision Tree classifier with a Streamlit frontend

Live App: https://loan--approval.streamlit.app/

---

## Project Structure

```
Bank_loan_auth/
├── bank3.py                   # Streamlit app and inference logic
├── model.pkl                  # Serialized DecisionTreeClassifier
├── upd_bank_loan (1).ipynb    # EDA training and evaluation notebook
└── requirements.txt           # Dependencies
```

---

## How It Works

### Input Features

| Feature | Type | Values |
|---|---|---|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Numerical | 0-10 |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| Credit_History | Numerical | 0.0 - 1.0 |
| Property_Area | Categorical | Rural / Semiurban / Urban |
| LoanAmount | Numerical | - |
| ApplicantIncome | Numerical | - |
| CoapplicantIncome | Numerical | - |
| Loan_Amount_Term | Numerical | Months |

### Preprocessing

Categorical encoding:
```python
gender_map    = {'Male': 0, 'Female': 1}
married_map   = {'No': 0, 'Yes': 1}
education_map = {'Graduate': 0, 'Not Graduate': 1}
employed_map  = {'No': 0, 'Yes': 1}
property_map  = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
```

Log transformation on skewed financial features:
```python
LoanAmount        = log(LoanAmount + 1)
ApplicantIncome   = log(ApplicantIncome + 1)
CoapplicantIncome = log(CoapplicantIncome + 1)
Loan_Amount_Term  = log(Loan_Amount_Term + 1)
Total_Income      = log(ApplicantIncome + CoapplicantIncome + 1)
```

Total_Income is a derived feature representing combined household repayment capacity

Output: 1 = Approved | 0 = Rejected

---

## Run Locally

```bash
git clone https://github.com/dishita-b/Bank_loan_auth.git
cd Bank_loan_auth
pip install -r requirements.txt
streamlit run bank3.py
```

App runs at http://localhost:8501

---

## Dependencies

```
joblib==1.3.2
numpy==1.26.4
pandas==2.2.1
Requests==2.31.0
scikit_learn==1.4.1.post1
streamlit==1.32.2
```

## Author

Dishita Barman
LinkedIn: https://www.linkedin.com/in/dishita-barman5/
GitHub: https://github.com/dishita-b
