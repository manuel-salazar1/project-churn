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







