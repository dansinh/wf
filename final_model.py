#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0. Setup
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler, FunctionTransformer, SplineTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, log_loss, recall_score, precision_score

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cols = [
    "id", "loan_amnt", "loan_status",
    "term", "int_rate", "installment", "emp_length",
    "home_ownership", "annual_inc", "purpose", "dti",
    "delinq_2yrs", "mths_since_last_delinq",
    "open_acc", "total_acc",
    "revol_bal", "earliest_cr_line"
]

df = (
    pd.read_csv("loan_data.csv")[cols]
    .rename(columns = {
        'id': 'customer_id',
        'loan_amnt': 'loan_amount',
        'int_rate': 'interest_rate',
        'emp_length': 'employment_length',
        'annual_inc': 'annual_income',
        'delinq_2yrs': 'delinquency_2years',
        'mths_since_last_delinq': 'months_since_delinquency',
        'earliest_cr_line': 'min_date_credit_line',
        'open_acc': 'open_accounts',
        'total_acc': 'total_accounts',
        'revol_bal': 'revolving_balance'
    })
    .assign(
        months_since_delinquency = lambda df_: 
            df_.months_since_delinquency.fillna(0)
    )
    .assign(
        loan_status = lambda df_:
            df_.loan_status.replace({
                'Charged Off': 'bad',
                'Default': 'bad',
                'Late (31-120 days)': 'bad',
                'Late (16-30 days)': 'bad',
                'In Grace Period': 'good',
                'Current': 'good',
                'Fully Paid': 'good'
            }).astype('category'),
        home_ownership = lambda df_: df_.home_ownership.replace({
            'MORTGAGE': 'mortgage',
            'RENT': 'rent',
            'OWN': 'own',
            'OTHER': 'mortgage',
            'NONE': 'mortgage' 
        }).astype('category')
    )
    .assign(
        employment_length = lambda df_:
            df_.employment_length.replace('< 1 year', 0.5)
            .str.extract(r'(\d+)')
            .astype('float')
            .fillna(0.5)
    )
    .assign(
        loan_amount = lambda df_: df_.loan_amount.astype('float'),
        term = lambda df_: df_.term.astype('category'),
        purpose = lambda df_: df_.purpose.astype('category'),
        min_date_credit_line = lambda df_: pd.to_datetime(df_.min_date_credit_line)
    )
    .assign(
        min_date_credit_line = lambda df_:
            np.where(
                df_['min_date_credit_line'] <= pd.Timestamp.today(),
                df_.min_date_credit_line,
                pd.to_datetime(pd.Timestamp.today())
            ).astype('datetime64[ns]')
    )
    .assign(
        months_since_first_credit = lambda df_:
            ((pd.to_datetime('today') - df_.min_date_credit_line) / np.timedelta64(1, 'M'))
    )
    .assign(
        loan_income_ratio = lambda df_:
            df_.loan_amount / df_.annual_income,
        installment_balance_ratio = lambda df_: 
            df_.installment / (df_.revolving_balance + 1),
        open_account_ratio = lambda df_: df_.open_accounts / df_.total_accounts,
        total_earnings = lambda df_: df_.employment_length * df_.annual_income
    )
    .dropna()
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
df.info()
