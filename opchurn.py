# All function for Telco Churn project, Final presentation

# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


####### EXPLORE #######

# Monthly charges and churn plot

def get_monthly_charges_plot(train):
    # isolating active customers for ttest
    no_churn = train[train.churn == 'No']
    
    # isolating churned customers for ttest
    churned = train[train.churn == 'Yes']
    
    # showing distribution of customers who have churned based on monthly charges
    plt.title('Monthly Charges and Customer Status')
    plt.hist(no_churn.monthly_charges, label='Not Churned')
    plt.hist(churned.monthly_charges, label='Churned')
    plt.xlabel('Monthly Charge Amount')
    plt.ylabel('Number of Customers')
    plt.legend()
    plt.show()

# -------------------------------------------------------------------------------------


# Monthly charges and churn t-test

def get_monthly_charges_ttest(train):
    # setting alpha (confidence level)
    alpha = 0.05
    
    # isolating churned customers for ttest
    churned = train[train.churn == 'Yes']
    
    # calculating mean monthly charges for ttest
    overall_mean = train.monthly_charges.mean()
    
    # initiating ttest
    t, p = stats.ttest_1samp(churned.monthly_charges, overall_mean)
    
    print(f't = {t:.4}')
    print(f'p = {p/2:.4}')
    
    # printing test outcome results
    if p/2 > alpha:
        print('We fail to reject the null hypothesis')
    elif t < 0:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the hypothesis')


# -------------------------------------------------------------------------------------


# Internet service type and churn plot

def get_internet_service_plot(train):
    # create observed for chi^2 test
    observed = pd.crosstab(train.internet_service_type, train.churn)
    
    # plot internet service type from crosstab
    observed.plot.bar(subplots=False, figsize=(6, 6), legend=True)
    plt.title('Internet Service')
    plt.xticks(rotation=0)
    plt.xlabel('Internet Service Type')
    plt.ylabel('Number of Customers')
    plt.show()


# -------------------------------------------------------------------------------------


# Internet service type and churn chi^2 test 

def get_internet_service_chi2(train):

    #set alpha
    alpha = 0.05

    # create observed for test
    observed = pd.crosstab(train.internet_service_type, train.churn)

    # setting up chi^2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # print the chi2 value, formatted to a float with 4 digits
    print(f'chi^2 = {chi2:.4f}')
    
    # print the p-value, formatted to a float with 4 digits
    print(f'p.    = {p:.4f}')
    
    # print the result of the test
    if p > alpha:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the null hypothesis')



# -------------------------------------------------------------------------------------

# Tenure and churn t-test

def get_tenure_ttest(train):
    
    # setting alpha
    alpha = 0.05
    
    # isolating churned and no churn customers for ttest
    churned = train[train.churn == 'Yes']
    no_churn = train[train.churn == 'No']
    
    # initiating variance test result is unequal variance
    stat, pval = stats.levene(churned.tenure, no_churn.tenure)
    stat, pval
    
    # initiating t-test
    t, p = stats.ttest_ind(churned.tenure, no_churn.tenure, equal_var=False)
    print(f't= {t:.4}')
    print(f'p= {p:.4}')
    
    if p > alpha:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the null hypothesis')



# -------------------------------------------------------------------------------------

# Contract type and churn chi^2 test

def get_contract_type_chi2(train):
    # set alpha
    alpha = 0.05
    
    #create observed for chi^2 test
    observed = pd.crosstab(train.contract_type, train.churn)
    
    # initiate chi^2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    
    # print the chi2 value, formatted to a float with 4 digits
    print(f'chi^2 = {chi2:.4f}')
    
    # print the p-value, formatted to a float with 4 digits
    print(f'p.    = {p:.4f}')
    
    if p > alpha:
        print('We fail to reject the null hypothesis')
    else:
        print('We reject the null hypothesis')




# -------------------------------------------------------------------------------------



######## MODELING ########


# create variable for columns I want to drop for model
# drop_cols is a global variable and will be called in for xy_train_val_test function

drop_cols = ['senior_citizen', 'total_charges', 'gender_Male', 'partner_Yes', 'dependents_Yes', 'phone_service_Yes'
            , 'multiple_lines_No phone service', 'multiple_lines_Yes', 'online_security_No internet service'
            , 'online_security_Yes', 'online_backup_No internet service', 'online_backup_Yes'
            , 'device_protection_No internet service', 'device_protection_Yes', 'streaming_tv_No internet service'
            , 'streaming_tv_Yes', 'streaming_movies_No internet service', 'streaming_movies_Yes'
            , 'paperless_billing_Yes', 'payment_type_Credit card (automatic)'
            , 'payment_type_Electronic check', 'payment_type_Mailed check']



# create function to initiate X_y train, validate, test
def Xy_train_val_test(train, validate, test, target_variable):
    """
    input train, validate, test, after using split function()
    input target_variable as string
    drop_cols formatted as: ['col1', 'col2', 'etc'] for multiple columns
        This function will drop all 'object' columns. Identify additional 
        columns you want to drop and insert 1 column as a string or multiple
        columns in a list of strings.
    X_train, X_validate, X_test, y_train, y_validate, y_test
    """
    
    baseline_accuracy = train[target_variable].value_counts().max() / train[target_variable].value_counts().sum()
    print(f'Baseline Accuracy: {baseline_accuracy:.2%}')
    
    X_train = train.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_validate = validate.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    X_test = test.select_dtypes(exclude=['object']).drop(columns=[target_variable]).drop(columns=drop_cols)
    
    y_train = train[target_variable]
    y_validate = validate[target_variable]
    y_test = test[target_variable]
    
    return X_train, X_validate, X_test, y_train, y_validate, y_test




# -------------------------------------------------------------------------------------


# Best random forest model function

def rand_forest_model(X_train, y_train, X_validate, y_validate):
    # best model from multiple iterations
    rf = RandomForestClassifier(random_state=123, min_samples_leaf=3, max_depth=8)
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_validate, y_validate)
    print(f'   Train Accuracy: {train_acc:.2%}')
    print(f'Validate Accuracy: {val_acc:.2%}')





# -------------------------------------------------------------------------------------


# Best decision tree model function

def decision_tree_model(X_train, y_train, X_validate, y_validate):
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    
    train_acc2 = clf.score(X_train, y_train)
    val_acc2 = clf.score(X_validate, y_validate)
    print(f'   Train Accuracy: {train_acc2:.2%}')
    print(f'Validate Accuracy: {val_acc2:.2%}')





# -------------------------------------------------------------------------------------

# Best logistic regression model function

def log_reg_model(X_train, y_train, X_validate, y_validate):
    logit = LogisticRegression()
    logit.fit(X_train, y_train)
    
    train_acc3 = logit.score(X_train, y_train)
    val_acc3 = logit.score(X_validate, y_validate)
    print(f'   Train Accuracy: {train_acc3:.2%}')
    print(f'Validate Accuracy: {val_acc3:.2%}')




# -------------------------------------------------------------------------------------

# Best model function


def best_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    rf = RandomForestClassifier(random_state=123, min_samples_leaf=3, max_depth=8)
    rf.fit(X_train, y_train)
    
    train_acc = rf.score(X_train, y_train)
    val_acc = rf.score(X_validate, y_validate)
    test_acc = rf.score(X_test, y_test)
    print('Baseline Accuracy: 73.47%')
    print(f'   Train Accuracy: {train_acc:.2%}')
    print(f'Validate Accuracy: {val_acc:.2%}')
    print(f'   Test Accuracry: {test_acc:.2%}')
    return rf




# -------------------------------------------------------------------------------------

# Create csv function

def create_csv(X_test, test, rf):
    """
    Export the predictions and prediction probabilities of a given test set into a CSV file.
    """
    # Set up CSV
    prediction = rf.predict(X_test)
    prediction_prob = rf.predict_proba(X_test)
    # Combine customer_id, prediction, and prediction_prob into a pandas DataFrame
    result_df = pd.DataFrame({
    'customer_id': test.customer_id,
    'prediction_prob_1': prediction_prob[:, 1],
    'prediction': prediction,})
    # Export the DataFrame as a CSV file
    result_df.to_csv('result.csv', index=False)
    return







