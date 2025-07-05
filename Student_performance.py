# first step import library needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# second step read data file my file csv format
student_data = pd.read_csv("Student.csv")
print(student_data.head(6))
# print(student_data.isnull().sum())
# clean data from empty values
student_data_2 = student_data.dropna()
X = student_data_2[['Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']]
Y = student_data_2['GPA']
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)*100
r2 = r2_score(y_test, y_pred)*100
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")