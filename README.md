# Project-churn

## Project description

Telco is a telecommunications company and they have requested our help to identify drivers for customer churn. They have provided me with a database (Codeup's DS telco_churn db) to facilitate this task. I will construct 3 ML classification models and at the end, pick the one that most accurately predicts customer churn based on the data provided. 

## Project goals:
- Identify key drivers for customer churn
- Use drivers to develop a ML model that accurately predicts whether a customer has churned
- Provide Telco with key customer insights
- Provide at least one business recommendation which Telco can use to meet their business objectives


## Initial hypotheses and/or questions you have of the data, ideas

- My initial hypotheses is that customer churn is dependent on the monthly charges, type of internet service, and bundled services, and promotions.

## data dictionary

| Feature | Definition |
|:--------|:-----------|
|Monthly Charges| The amount of money in USD a customer is charged per month (float)|
|Tenure| How long a customer has been with Telco (months)|
|Churn| A customer has left Telco|
|No Churn| An active Telco customer|
|Internet Service Type| Whether a customer has DSL, Fiber optic, or None|
|Contract type| Whether a customer has a Month-to-month, One-year, or Two-year contract|
|EDA| Exploritory Data Analysis|


## Project planning (lay out your process through the data science pipeline)

#### Aquire data from Codeup DS telco_churn db
 
#### Prepare data

Identify null values (if there are, explain how I handled null values)
Create Engineered columns from existing data if time permits:
bundled (if a customer has multiple products or product features)

#### Explore data
Explore data in search of drivers of churn

Answer the following questions(min. first 4):
Does monthly charges effect customer churn?
Does internet service type effect customer churn?
Does tenure effect customer churn?
Does contract type effect customer churn?
Does having tech support effect customer churn?
Does having bundled services effect customer churn?
Does age effect customer churn?


#### Develop a Model to predict if a customer will churn based on key drivers.

Use drivers identified in explore to build predictive models
Evaluate models on train and validate data
Select the best model based on highest accuracy for each algorithm
Evaluate the best model on test data

#### Draw conclusions

## Steps to reproduce

Clone this repo.
You will need your own env file with Codeup database credentials along with all the necessary files listed below to run my final project notebook.

 Read this README.md
 Download the aquire.py, prepare.py, opchurn.py and final_report.ipynb files into your working directory
 Add your own env file to your directory. (user_name, password, host)
 Run the final_report.ipynb notebook


## Recommendations, key findings, and takeaways from this project

1. Revisit Telco value proposition:

- The cost of maintaining a customer is typically cheaper than the cost of acquiring a customer.
    - Retaining customers for longer will increase the bottom line
- Explore additional add ons or subsidized subscription services for:
    - New customers AND
    - Tenured customers
        - Both offers should be different with a perceived higher value for more tenured customers to extend their lifetime value.
- Lean in and increase our competitive advantage.

2. Dive into the competitor landscape to understand how that is affecting Telco:

- Is there a new competitor that our customers are jumping to?
- Are competitors offering the same, less or more services for the same, lower, or higher price? (value proposition)
- Do Telco competitors have features that Telco doesn't offer?
    - If yes, can Telco incorporate similar features at low cost and minimal interruption to the current workforce?
- Price is typically elastic in the telecommunications industry but don ot make it a race to the bottom. There are other ways Telco can incentivize customers to stay longer.

3. Take a look inside the company:

- An overwhelming majority of customers who churn have fiber internet and are paying higher monthly charges than the average customer.
    - Are there service interruptions for fiber service?
    - Is our fiber service speed underperforming advertised contract speeds?
        - If yes to either, does Telco have the proper workforce in place to minimize interruptions or expedite maintenance fix issues?


- Monthly to month contracts naturally have higher churn rates.
- This industry has low entry and exit barriers which means it's easy for customer to switch back and forth between Telco and its competitors.


## Next Steps
If I had more time I would have wanted to create a column of bundled service or add-ons and test the significance that variable would have on customer churn. My hypothesis on this is that customers who have a deeper relationship (more services) churn at a lower rate.


