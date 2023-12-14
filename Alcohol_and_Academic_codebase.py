# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:23:02 2023

Using "Dataset on Student Alcoholism and Academic Performance in High School" to run some models
Dataset sourced from Kaggle

Citation:
    P. Cortez e A. Silva. Usando a Mineração de Dados para Prever o Desempenho do Aluno do Ensino Médio. Em A. Brito e J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008) pp. 5-12, Porto, Portugal, abril de 2008, EUROSIS, ISBN 978-9077381-39-7.


@author: Juan
"""

import pandas as pd

from sklearn.preprocessing import LabelEncoder 
#labelencoder seems to be quite problematic, turns out will not use labelencoder

# importing those models that i am intending to run (just those few that i know as of now)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

file_path = "Dataset on Student Alcoholism and Academic Performance in High School/en_lpor_explorer.csv"
student_data_raw = pd.read_csv(file_path)

student_data = pd.read_csv(file_path)

student_data.describe()
student_data.columns


## more preprocessing work for features with categories
student_data = pd.get_dummies(student_data, columns = ["Health_Status",
                                                       "Weekly_Study_Time",
                                                       "Legal_Responsibility",
                                                       "Desire_Graduate_Education",
                                                       "Is_Dating"])


student_data_filtered = student_data.dropna(axis = 0) #remove the students who have incomplete data
## seems like all the data were complete to begin with, very clean dataset we have here


# decide what should our y variable or dependent variable should be?
## it can be either Grade_1st_Semester or Grade_2nd_Semester
## the students are graded from range 0-20, not sure 0 or 20 is best..
## guess have to find out a bit of Portugese high school grading context

y1 = student_data_filtered.Grade_1st_Semester # lets go with y1 for a start
y2 = student_data_filtered.Grade_2nd_Semester

# decide on what X variables or features to include
print(student_data.columns)
features = ["Health_Status_Very Poor","Health_Status_Poor","Health_Status_Good","Health_Status_Very Good",
            "School_Absence",
            "Weekly_Study_Time_2 to 5h","Weekly_Study_Time_5 to 10h","Weekly_Study_Time_More than 10h",
            "Legal_Responsibility_Father","Legal_Responsibility_Mother",
            "Desire_Graduate_Education_Yes",
            "Is_Dating_Yes"]
X = student_data_filtered[features]

# split the data between training & testing sets
train_X, test_X, train_y1, test_y1 = train_test_split(X, y1, random_state = 1)

# running the linear model
student_linear_model = LinearRegression()
student_linear_model.fit(train_X,train_y1)
# get the coefficients
coefficients = student_linear_model.coef_
coefficients_df = pd.DataFrame({"Feature":X.columns, "Coefficients":coefficients})
print(coefficients_df)
# predicting the test data
linear_test_predictions = student_linear_model.predict(test_X)
linear_mae = mean_absolute_error(test_y1, linear_test_predictions)
print(f"The Linear model produces a mean absolute error of {linear_mae}")


# running the DecisionTree model
student_tree_model = DecisionTreeRegressor(random_state = 1)
student_tree_model.fit(train_X,train_y1)
# predict the test data
tree_test_predictions = student_tree_model.predict(test_X)
tree_mae = mean_absolute_error(test_y1, tree_test_predictions)
print(f"The DecisionTree model produces a mean absolute error of {tree_mae}")


# running the RandomForest model
student_rf_model = RandomForestRegressor(random_state = 1)
student_rf_model.fit(train_X,train_y1)
# predict the test data
rf_predictions = student_rf_model.predict(test_X)
rf_mae = mean_absolute_error(test_y1, rf_predictions)
print(f"The RandomForest model produces a mean absolute error of {rf_mae}")



# well, a bummer when the simple linear model seems to have the lowest mae
# lets create two models using all the data, linear and random forest models
## this is only useful if we have a new set of data to predict
## otherwise, we can just end here

## 1st model: Linear model
student_linear = LinearRegression()
student_linear.fit(X,y1)


## 2nd model: Random forest model
student_rf = RandomForestRegressor(random_state = 1)
student_rf.fit(X,y1)
