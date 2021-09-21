# import warnings
# warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import statsmodels.formula.api as smf

url = 'https://raw.githubusercontent.com/rusita-ai/pyData/master/Insurance.csv'
DF = pd.read_csv(url)

# distribution of expenses
plt.figure(figsize = (9, 6))
sns.distplot(DF.expenses,
             hist = True,
             kde = True)
plt.show()

# boxplot of expenses
plt.figure(figsize = (9, 6))
sns.boxplot(y = 'expenses', data = DF)
plt.show()

# sex vs expenses
plt.figure(figsize = (9, 6))
sns.boxplot(x = 'sex', y = 'expenses', data = DF)
plt.show()

# children vs expenses
plt.figure(figsize = (9, 6))
sns.boxplot(x = 'children', y = 'expenses', data = DF)
plt.show()
# DF.children.value_counts()

# smoker vs expenses
plt.figure(figsize = (9, 6))
sns.boxplot(x = 'smoker', y = 'expenses', data = DF)
plt.show()

# region vs expenses
plt.figure(figsize = (9, 6))
sns.boxplot(x = 'region', y = 'expenses', data = DF)
plt.show()

# bmi distributiton
plt.figure(figsize = (9, 6))
sns.distplot(DF.bmi,
             hist = True,
             kde = True)
plt.show()

# bmi vs expenses
plt.figure(figsize = (9, 6))
sns.scatterplot(x = DF.bmi, y = DF.expenses)
plt.show()

from sklearn.preprocessing import LabelEncoder

# integer encoding (== label encoding)
encoder1 = LabelEncoder()
DF['sex'] = encoder1.fit_transform(DF.sex)
encoder2 = LabelEncoder()
DF['smoker'] = encoder2.fit_transform(DF.smoker)
encoder3 = LabelEncoder()
DF['region'] = encoder3.fit_transform(DF.region)



# train/test - train_test_split()
X = DF[['age', 'sex']]
y = DF['expenses']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 2045)

# LinearRegression(), .fit()
RA = LinearRegression()
RA.fit(X_train, y_train)
# .predict
y_hat = RA.predict(X_test)
# MSE (note. difference from variance)
mse2 = mean_squared_error(y_test, y_hat)
#np.sqrt(mse2)



# train/test - train_test_split()
train_set, test_set = train_test_split(DF,
                                       test_size = 0.3,
                                       random_state = 2045)

# generate Model_1 by train_set
Model_1 = smf.ols(formula = 'expenses ~ age + sex',
                  data = train_set).fit()

# .predict()
y_hat_1 = Model_1.predict(test_set[['age', 'sex']])

# MSE
mse1 = mean_squared_error(test_set.expenses, y_hat_1)
np.sqrt(mse1)

# there is little difference between MSE of sklearn & OLS of statsmodels
print('statsmodels :', np.sqrt(mse1))
print('sklearn     :', np.sqrt(mse2))