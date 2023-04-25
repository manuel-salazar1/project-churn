import pandas as pd 
from sklearn.model_selection import train_test_split
import os





#Prep/clean functions

#IRIS

def prep_iris(iris):
    '''
    - This function will clean the iris dataset
    '''
    iris = iris.drop(columns=['species_id', 'measurement_id'])
    
    iris = iris.rename(columns={'species_name': 'species'})
    
    dummy_iris = pd.get_dummies(iris[['species']], drop_first=True)
    iris = pd.concat([iris, dummy_iris], axis=1)
    return iris


#TITANIC

def prep_titanic(titanic):
    '''
    - This function will clean the titanic dataset
    '''
    titanic = titanic.drop(columns=['embark_town', 'class', 'deck']) #, 'age'
    
    titanic.embarked = titanic.embarked.fillna(value='S')
    
    dummy_titanic = pd.get_dummies(titanic[['sex', 'embarked']], drop_first=True)
    titanic = pd.concat([titanic, dummy_titanic], axis=1)
    return titanic


#TELCO

def prep_telco(telco):
    '''
    This function will clean the telco dataset
    '''
    telco = telco.drop(columns=['payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    
    telco.total_charges = telco.total_charges.str.replace(' ', '0').astype(float)
    
    dummy_telco = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                      'phone_service', 'multiple_lines', 
                      'online_security', 'online_backup', 
                      'device_protection', 'tech_support', 
                      'streaming_tv', 'streaming_movies', 
                      'paperless_billing', 'churn', 'contract_type', 
                      'internet_service_type', 'payment_type']], drop_first=True)
    telco = pd.concat([telco, dummy_telco], axis=1)
    return telco





#SPLIT FUNCTION

def split_function(df, target_variable):
    '''
    Take in a data frame and returns train, validate, test subset data frames
    Input target_variable as a string
    '''
    train, test = train_test_split(df,
                              test_size=0.20,
                              random_state=123,
                              stratify= df[target_variable]
                                  )
    train, validate = train_test_split(train,
                                  test_size=.25,
                                  random_state=123,
                                   stratify= train[target_variable]
                                      )
    return train, validate, test


