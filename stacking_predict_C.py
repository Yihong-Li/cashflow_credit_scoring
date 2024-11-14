import zipfile
import os
import numpy as np
import pandas as pd
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.model_selection as ms
import pickle
import datetime
from sklearn.ensemble import HistGradientBoostingClassifier

def predict(data_paths, pkl_path, ohe=None):
	#get data
	consumers = pd.read_parquet(data_paths + '/consumer_data.parquet')
	transactions = pd.read_parquet(data_paths + '/transactions.parquet')
	categories = pd.read_csv(data_paths + '/transaction_categories.csv')

	#Feature engineering
	transactions = pd.merge(transactions, categories, left_on='category', right_on='code', how='left')
	merged_df = pd.merge(transactions, consumers, left_on='masked_consumer_id', right_on='masked_consumer_id', how='left')
	for col in merged_df.select_dtypes(include=['float64']).columns:
		merged_df[col] = merged_df[col].astype('float32')
	merged_df['ClientID'] = merged_df['masked_consumer_id'].str[:3]

	# Map ClientID to product type
	client_product_map = {
		'C01': 'Personal Loans',
		'C02': 'Payday Loans',
		'C03': 'Credit Cards',
		'C04': 'Personal Loans'
	}

	merged_df['Product'] = merged_df['ClientID'].map(client_product_map)
	merged_df = merged_df.drop(columns=['ClientID', 'masked_transaction_id', 'code', 'description'])
	merged_df['posted_date'] = pd.to_datetime(merged_df['posted_date'])
	merged_df['posted_year'] = merged_df['posted_date'].dt.year
	category_mapping = dict()
	for row in categories.iterrows():
		category_mapping[row[1]['code']] = row[1]['description']

	merged_df = merged_df.dropna()
	# Convert 'posted_date' to datetime format
	merged_df['posted_date'] = pd.to_datetime(merged_df['posted_date'])
	merged_df['evaluation_date'] = pd.to_datetime(merged_df['evaluation_date'])

	consumer_features1 = merged_df.groupby('masked_consumer_id').agg(
		transactions_count=('masked_consumer_id', 'count'),
		max_debit_amount=('amount', lambda x: x[x < 0].max()),
		min_debit_amount=('amount', lambda x: x[x < 0].min()),
		average_debit_amount=('amount', lambda x: x[x < 0].mean()),
		max_credit_amount=('amount', lambda x: x[x > 0].max()),
		min_credit_amount=('amount', lambda x: x[x > 0].min()),
		average_credit_amount=('amount', lambda x: x[x > 0].mean()),
		count_of_credits=('amount', lambda x: x[x > 0].count()),
		count_of_debits=('amount', lambda x: x[x < 0].count()),		
	).reset_index()

	# Fill NaN values with 0
	consumer_features1.fillna(0, inplace=True)
	consumer_features2 = merged_df[['masked_consumer_id', 'FPF_TARGET', 'evaluation_date', 'total_balance']].drop_duplicates()

	# Calculate days span for each consumer
	merged_df['days_span'] = (merged_df['evaluation_date'] - merged_df.groupby('masked_consumer_id')['posted_date'].transform('min')).dt.days.clip(lower=1)

	# Calculate total debit and credit sums for each consumer
	debit_credit_sums = merged_df.groupby('masked_consumer_id').apply(
		lambda df: pd.Series({
			'total_debit': df[df['amount'] < 0]['amount'].sum(),
			'total_credit': df[df['amount'] > 0]['amount'].sum(),
			'transaction_count': df['posted_date'].count(),
			'days_span': df['days_span'].iloc[0]
		})
	)

	# Calculate averages for debits, credits, and transaction frequency
	debit_credit_sums['average_debit_spending'] = debit_credit_sums['total_debit'] / debit_credit_sums['days_span']
	debit_credit_sums['average_credit_spending'] = debit_credit_sums['total_credit'] / debit_credit_sums['days_span']
	debit_credit_sums['average_transaction_frequency'] = debit_credit_sums['transaction_count'] / debit_credit_sums['days_span']

	# Calculate total amounts for specific categories
	for code, cat in category_mapping.items():
		category_sum = merged_df[merged_df['category'] == code].groupby('masked_consumer_id')['amount'].sum()
		debit_credit_sums[f'total_{cat}'] = category_sum

	# Calculate average amounts for specific categories
	for cat in category_mapping.values():
		debit_credit_sums[f'average_{cat}'] = debit_credit_sums[f'total_{cat}'] / debit_credit_sums['days_span']
	debit_credit_sums.fillna(0, inplace=True)
	debit_credit_sums.reset_index(inplace=True)
	consumer_data = consumer_features1.merge(consumer_features2, on='masked_consumer_id')
	consumer_data = consumer_data.merge(debit_credit_sums, on='masked_consumer_id')

	cols_to_drop = [col for col in consumer_data.columns if col.startswith('total_')]
	consumer_data = consumer_data.drop(columns=cols_to_drop)
	working_data = consumer_data.copy()
	# working_data['credit_to_debit_ratio'] = working_data['average_credit_amount'] / working_data['average_debit_amount'].abs()
	working_data['investment_to_income_ratio'] = working_data['average_INVESTMENT_INCOME'] / (working_data['average_PAYCHECK'] + 1e-5)  # Adding a small number to avoid division by zero
	working_data['debit_volatility'] = working_data.groupby('masked_consumer_id')['average_debit_amount'].transform(np.std)
	working_data['credit_volatility'] = working_data.groupby('masked_consumer_id')['average_credit_amount'].transform(np.std)
	working_data['max_min_debit_ratio'] = working_data['max_debit_amount'] / (working_data['min_debit_amount'].abs() + 1e-5)  # To avoid division by zero
	working_data['high_cost_debit_flag'] = (working_data['max_debit_amount'] < -5000).astype(int)  # High cost debit transaction
	working_data['low_cost_credit_flag'] = (working_data['min_credit_amount'] < 100).astype(int)  # Low cost credit transaction
	working_data['transactions_per_day'] = working_data['transactions_count'] / working_data['days_span']
	working_data['cumulative_credits'] = working_data.groupby('masked_consumer_id')['average_credit_amount'].cumsum()
	working_data['cumulative_debits'] = working_data.groupby('masked_consumer_id')['average_debit_amount'].cumsum()
	working_data['has_education_transactions'] = (working_data['average_EDUCATION'] != 0).astype(int)
	working_data['log_max_credit'] = np.log(working_data['max_credit_amount'] + 1)  # Log of max credit amount to reduce skewness
	working_data['log_max_debit'] = np.log(working_data['max_debit_amount'].abs() + 1)  # Log of absolute max debit amount to reduce skewness
	working_data['extreme_healthcare_spending'] = (working_data['average_HEALTHCARE_MEDICAL'] < -100).astype(int)
	working_data['extreme_education_spending'] = (working_data['average_EDUCATION'] < -100).astype(int)
	working_data['proportion_credits'] = working_data['count_of_credits'] / working_data['transactions_count']
	working_data['proportion_debits'] = working_data['count_of_debits'] / working_data['transactions_count']
	# Standard deviation of transaction amounts as an indicator of financial stability
	working_data['std_transaction_amount'] = working_data[['average_debit_amount', 'average_credit_amount']].std(axis=1)
	# Ratio of total credits to total debits to indicate spending efficiency
	working_data['spending_efficiency'] = working_data['average_credit_amount'] / (working_data['average_debit_amount'].abs() + 1e-5)
	# Assuming 'evaluation_date' is in YYYY-MM-DD format and you want to capture seasonal effects
	working_data['month'] = pd.to_datetime(working_data['evaluation_date']).dt.month
	working_data['is_high_spending_season'] = working_data['month'].isin([11, 12]).astype(int)  # Nov and Dec as high spending months
	# Measure variability in transaction frequency to assess stability
	working_data['frequency_stability'] = working_data.groupby('masked_consumer_id')['transactions_per_day'].transform(np.std)
	# Flags for unusual spending or income patterns
	working_data['high_credit_frequency'] = (working_data['count_of_credits'] > working_data['count_of_credits'].quantile(0.95)).astype(int)
	working_data['low_debit_frequency'] = (working_data['count_of_debits'] < working_data['count_of_debits'].quantile(0.05)).astype(int)
	working_data2 = working_data.copy()
	# working_data2['CLV'] = working_data2.groupby('masked_consumer_id').cumsum()['average_credit_amount'] - working_data2.groupby('masked_consumer_id').cumsum()['average_debit_amount']
	# Create rolling averages for credits and debits to capture trends over time
	working_data2.sort_values(by=['masked_consumer_id', 'evaluation_date'], inplace=True)  # Ensure data is sorted
	working_data2['rolling_avg_credits'] = working_data2.groupby('masked_consumer_id')['average_credit_amount'].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
	working_data2['rolling_avg_debits'] = working_data2.groupby('masked_consumer_id')['average_debit_amount'].rolling(window=12, min_periods=1).mean().reset_index(level=0, drop=True)
	# Extract day of week from date and create dummy variables for weekdays
	working_data2['day_of_week'] = pd.to_datetime(working_data2['evaluation_date']).dt.dayofweek
	working_data2 = pd.get_dummies(working_data2, columns=['day_of_week'], prefix='dow')
	# Create an interaction term between credits and debits to explore their combined effect on the target variable
	working_data2['credit_debit_interaction'] = working_data2['average_credit_amount'] * working_data2['average_debit_amount']
	# Create lag features for credit and debit amounts to capture previous transaction values
	working_data2['lag1_credit'] = working_data2.groupby('masked_consumer_id')['average_credit_amount'].shift(1)
	working_data2['lag1_debit'] = working_data2.groupby('masked_consumer_id')['average_debit_amount'].shift(1)
	# Calculate the percentage change from the previous record to capture growth or contraction
	working_data2['pct_change_credit'] = working_data2.groupby('masked_consumer_id')['average_credit_amount'].pct_change()
	working_data2['pct_change_debit'] = working_data2.groupby('masked_consumer_id')['average_debit_amount'].pct_change()
	# Flags to identify unusually high or low transactions
	working_data2['extreme_high_credit'] = (working_data2['average_credit_amount'] > working_data2['average_credit_amount'].quantile(0.99)).astype(int)
	working_data2['extreme_low_debit'] = (working_data2['average_debit_amount'] < working_data2['average_debit_amount'].quantile(0.01)).astype(int)

	spx500 = working_data2.copy()
	spx500['Essential_vs_NonEssential'] = spx500[['average_FOOD_AND_BEVERAGES', 'average_GROCERIES', 'average_HEALTHCARE_MEDICAL']].sum(axis=1) / spx500[['average_ENTERTAINMENT', 'average_TRAVEL', 'average_HOME_IMPROVEMENT']].sum(axis=1)
	spx500['total_credit'] = spx500['average_credit_amount'] * spx500['count_of_credits']
	spx500['total_debit'] = spx500['average_debit_amount'] * spx500['count_of_debits']
	spx500['net_flow'] = spx500['total_credit'] - spx500['total_debit']
	spx500['credit_debit_ratio'] = spx500['total_credit'] / spx500['total_debit']
	# Spending Patterns
	spx500['entertainment_to_total_spending'] = spx500['average_ENTERTAINMENT'] / spx500['average_credit_spending']
	spx500['grocery_to_total_spending'] = spx500['average_GROCERIES'] / spx500['average_credit_spending']
	spx500['essential_spending'] = spx500['average_FOOD_AND_BEVERAGES'] + spx500['average_HEALTHCARE_MEDICAL'] + spx500['average_BILLS_UTILITIES']
	spx500['non_essential_spending'] = spx500['average_ENTERTAINMENT'] + spx500['average_TRAVEL'] + spx500['average_HOME_IMPROVEMENT']
	spx500['essential_to_non_essential_ratio'] = spx500['essential_spending'] / spx500['non_essential_spending']
	# Health & Insurance
	spx500['health_to_insurance_ratio'] = spx500['average_HEALTHCARE_MEDICAL'] / spx500['average_INSURANCE']
	spx500['insurance_spending_ratio'] = spx500['average_INSURANCE'] / spx500['total_credit']
	# Loans and Advances
	spx500['loan_to_income_ratio'] = (spx500['average_MORTGAGE'] + spx500['average_RENT'] + spx500['average_AUTO_LOAN']) / spx500['average_PAYCHECK']
	spx500['advance_to_deposit_ratio'] = spx500['average_SMALL_DOLLAR_ADVANCE'] / spx500['average_DEPOSIT']
	# Investment & Savings
	spx500['investment_to_income_ratio'] = spx500['average_INVESTMENT_INCOME'] / spx500['average_PAYCHECK']
	spx500['savings_index'] = spx500['average_DEPOSIT'] / spx500['total_credit']
	# Miscellaneous and Fees
	spx500['fees_to_total_transactions'] = spx500['average_ACCOUNT_FEES'] / (spx500['count_of_credits'] + spx500['count_of_debits'])
	spx500['miscellaneous_to_total_spending'] = spx500['average_MISCELLANEOUS'] / spx500['average_credit_spending']
	# Transfers and Refunds
	spx500['self_transfer_ratio'] = spx500['average_SELF_TRANSFER'] / spx500['total_credit']
	spx500['external_transfer_ratio'] = spx500['average_EXTERNAL_TRANSFER'] / spx500['total_credit']
	spx500['refund_to_spending_ratio'] = spx500['average_REFUND'] / spx500['average_credit_spending']
	# Additional ratios and comparisons
	for i in range(1, 11):
		spx500[f'credit_debit_ratio_{i}'] = spx500['average_credit_amount'] / spx500['average_debit_amount']
	# High Spending Flags
	spx500['high_spending_flag_groceries'] = (spx500['average_GROCERIES'] > spx500['average_GROCERIES'].quantile(0.75)).astype(int)
	spx500['high_spending_flag_entertainment'] = (spx500['average_ENTERTAINMENT'] > spx500['average_ENTERTAINMENT'].quantile(0.75)).astype(int)
	# Frequent Transactions Flags
	spx500['frequent_debits_flag'] = (spx500['count_of_debits'] > spx500['count_of_debits'].quantile(0.75)).astype(int)
	spx500['frequent_credits_flag'] = (spx500['count_of_credits'] > spx500['count_of_credits'].quantile(0.75)).astype(int)

	# Polynomial features to detect non-linear relationships
	spx500['credit_squared'] = spx500['average_credit_amount'] ** 2
	spx500['debit_cubed'] = spx500['average_debit_amount'] ** 3
	# Logarithmic transformations to normalize distributions
	spx500['log_credit'] = np.log1p(spx500['average_credit_amount'])
	spx500['log_debit'] = np.log1p(spx500['average_debit_amount'])
	# # Binned features to categorize continuous variables into discrete bins
	# spx500['credit_bins'] = pd.cut(spx500['average_credit_amount'], bins=4, labels=['Low', 'Medium', 'High', 'Very High'])
	# Rate of change to understand the velocity of financial metrics change
	spx500['credit_rate_change'] = spx500['average_credit_amount'].pct_change().fillna(0)

	spx500['credit_utilization'] = (spx500['average_CREDIT_CARD_PAYMENT'] + spx500['average_AUTO_LOAN']) / spx500['max_credit_amount']
	# Expense to income ratio: Higher ratios may suggest living beyond means.
	spx500['expense_to_income_ratio'] = (spx500['average_BILLS_UTILITIES'] + spx500['average_RENT'] + spx500['average_MORTGAGE']) / spx500['average_PAYCHECK']
	# Savings ratio: Proportion of deposits to paycheck, a higher ratio indicates better financial health.
	spx500['savings_ratio'] = spx500['average_DEPOSIT'] / spx500['average_PAYCHECK']
	spx500['credit_amount_cv'] = spx500['average_credit_amount'].std() / spx500['average_credit_amount'].mean()
	spx500['debit_amount_cv'] = spx500['average_debit_amount'].std() / spx500['average_debit_amount'].mean()
	# Variability in credit and debit transactions (measuring frequency changes)
	spx500['credit_transactions_cv'] = spx500['count_of_credits'].std() / spx500['count_of_credits'].mean()
	spx500['debit_transactions_cv'] = spx500['count_of_debits'].std() / spx500['count_of_debits'].mean()
	spx500['net_cash_flow'] = spx500['average_credit_amount'] - spx500['average_debit_amount']
	# Cash flow stability (std deviation of monthly net cash flow)
	spx500['cash_flow_stability'] = spx500[['average_credit_amount', 'average_debit_amount']].std(axis=1)
	# Total loan servicing (sum of all loan-related expenses)
	spx500['total_loan_servicing'] = spx500['average_MORTGAGE'] + spx500['average_RENT'] + spx500['average_AUTO_LOAN'] + spx500['average_CREDIT_CARD_PAYMENT']
	# Proportion of income spent on loans
	spx500['loan_servicing_ratio'] = spx500['total_loan_servicing'] / spx500['average_PAYCHECK']
	# Ratio of essential to total transactions (indicative of conservative spending)
	spx500['essential_transaction_ratio'] = (spx500['average_GROCERIES'] + spx500['average_BILLS_UTILITIES'] + spx500['average_HEALTHCARE_MEDICAL']) / (spx500['total_debit'] + spx500['total_credit'])
	# High risk transaction ratio (spending on non-essentials to essentials)
	spx500['high_risk_transaction_ratio'] = (spx500['average_ENTERTAINMENT'] + spx500['average_TRAVEL']) / (spx500['average_FOOD_AND_BEVERAGES'] + spx500['average_HEALTHCARE_MEDICAL'])
	# External transfer proportion (compared to total transactions)
	spx500['external_transfer_proportion'] = spx500['average_EXTERNAL_TRANSFER'] / (spx500['average_credit_amount'] + spx500['average_debit_amount'])
	# Dependency on external financial aid (indicates external financial support)
	spx500['financial_aid_dependency'] = spx500['average_OTHER_BENEFITS'] / spx500['average_PAYCHECK']

	# Replace infinity values with NaN

	spx500 = spx500.replace([np.inf, -np.inf, np.nan], 0)
	# spx500['Spending'] = pd.to_numeric(spx500['Spending'], errors='coerce')

	# Split data into features and target
	X = spx500.drop(columns=['evaluation_date', 'masked_consumer_id', 'FPF_TARGET'])
	y = spx500['FPF_TARGET']
	scaler = StandardScaler()
	X = scaler.fit_transform(X)
	# Load the model from the .pkl file
	with open(pkl_path, 'rb') as file:
		loaded_model = pickle.load(file)
		print('Model loaded successfully')
	y_pred_proba = loaded_model.predict_proba(X)[:, 1]
	# y_pred_df = pd.DataFrame(y_pred, columns=['predictions'])
	return y_pred_proba, y

def score(y_true, y_pred):
    """Cashflow scoring function."""
    return roc_auc_score(y_true, y_pred)


# Predict
data_path = ".../cashflow"   #note here should be a directory that contains the consumer_data.parquet, transactions.parquet, transaction_categories.csv
pkl_path = ".../stacking_model_C.pkl"   # this should be the directory to the model.pkl file
y_pred_proba, y_true = predict(data_path, pkl_path)

# Score
#   order of ids in consumer_data.parquet must match
#   order of ids returned from predict
print("Final AUC score is", score(y_true, y_pred_proba))

































