# fit(), predict(), train_test_split()

import warnings
warnings.filterwarnings('ignore')
# C:\ProgramData\Anaconda3\lib\site-packages\seaborn\_decorators.py:36:
# FutureWarning: Pass the following variables as keyword args: x, y.
# From version 0.12, the only valid positional argument will be `data`,
# and passing other arguments without an explicit keyword will result in an error or misinterpretation.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


url = 'https://raw.githubusercontent.com/rusita-ai/pyData/master/Electric.csv'
Elec = pd.read_csv(url)
Elec.info()
Elec.head()

sns.scatterplot(Elec['surface_area'], Elec['electricity'])
plt.show()

# 1st order regression
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'red'},
            scatter_kws={'edgecolor' : 'white'})
plt.xlim(505, 820)
plt.show()

# 2nd order regression
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'blue'},
            scatter_kws={'edgecolor' : 'white'},
            order=2)
plt.xlim(505, 820)
plt.show()

# 5rd order regression
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'green'},
            scatter_kws={'edgecolor' : 'white'},
            order=5)
plt.xlim(505, 820)
plt.show()


# 9th order regression
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'orange'},
            scatter_kws={'edgecolor' : 'white'},
            order=9)
plt.xlim(505, 820)
plt.ylim(50, 450)
plt.show()

# 비교
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'red'})
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'blue'},            
            order=2)
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'green'},
            order=5)
sns.regplot(x='surface_area', y='electricity', data=Elec,
            line_kws={'color':'orange'},
            order=9,
            scatter_kws={'color':'gray', 'edgecolor':'white'})
plt.xlim(505, 820)
plt.ylim(50, 450)
plt.xticks(rotation=35)
plt.yticks(rotation=90)
plt.show()


X_train = Elec[['surface_area']] # DataFrame
y_train = Elec['electricity'] # Series

Model_1 = LinearRegression()
Model_1.fit(X_train, y_train)


print(Model_1.coef_) # w
print(Model_1.intercept_) # b
y_hat_1 = Model_1.predict(X_train) # y_hat_1 == predicted values
len(y_hat_1)

TR_Err_1 = np.mean((y_train - y_hat_1) ** 2)
TR_Err_1

# x -> P_5(x)
poly = PolynomialFeatures(degree=5, include_bias=False)
PX_5 = poly.fit_transform(X_train)
PX_5

X_train.shape, PX_5.shape

from sklearn.linear_model import LinearRegression
Model_5 = LinearRegression()
Model_5.fit(PX_5, y_train)


