# first step import library needed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
student_data = pd.read_csv("Student.csv")
student_data_2 = student_data.dropna()
plt.figure(figsize=(10, 6))
#plt.plot(student_data['StudyTimeWeekly'], student_data['GPA'], marker='o', color='red')
plt.plot(student_data_2['Absences'], student_data_2['GPA'], marker='o', color='red')
plt.show()