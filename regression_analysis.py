# import warnings
# warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

DF = sns.load_dataset('mpg') #seaborn에서 제공해주는 test file인가?
DF.info()
print(DF.head(3))

DF1 = DF[['mpg', 'cylinders', 'displacement', 'weight']]
print(DF1.head(3))

# 그래프를 그려봄(산점도)
plt.figure(figsize=(9,6))
plt.scatter(x=DF1.weight, y=DF1.mpg, s=30)
plt.show()

# 같은 그래프를 조금 다른 문법으로 그려봄
fig = plt.figure(figsize=(9, 6))
sns.regplot(x='weight', y='mpg', data=DF1, fit_reg=False) # reg line은 안그림
plt.show()

# nxn pair plots
sns.pairplot(DF1)
plt.show()


stats.pearsonr(DF1.mpg, DF1.weight) # corr & P value
stats.pearsonr(DF1.mpg, DF1.displacement)
stats.pearsonr(DF1.mpg, DF1.cylinders)

X = DF1[['weight']]
y = DF1['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state=2045
                                                    )
# random_state=0 처럼 넣고 싶은거 넣으면 됨 그냥 seed임

print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)


RA = LinearRegression()
RA.fit(X_train, y_train)


print('weight(w) : ', RA.coef_)
print('bias(b) : ', RA.intercept_)

RA.score(X_test, y_test) # R^2 == R-Squre == 결정계수

y_hat = RA.predict(X_test)
mean_squared_error(y_test, y_hat)

y_hat1 = RA.predict(X)

plt.figure(figsize=(9, 6))
ax1 = sns.distplot(y, hist=False, label='y')
ax2 = sns.distplot(y_hat1, hist=False, label='y_hat', ax=ax1)
plt.ylim(0, 0.07)
plt.show()


DF2 = DF[['mpg', 'cylinders', 'horsepower', 'weight']]
DF2.head(3)

X = DF2[['weight']]
y = DF2['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2045)
print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)

## 선형 회귀 Modeling(다항회귀)
poly = PolynomialFeatures(degree=2, include_bias=False) # degree vs order
X_train_poly = poly.fit_transform(X_train)

print('변환 전 데이터: ', X_train.shape)
print('2차항 변환 데이터: ', X_train_poly.shape)
NL = LinearRegression()
NL.fit(X_train_poly, y_train)

# import numpy as np
# np.set_printoptions(suppress=True, precision=10)

print('weight(w) : ', NL.coef_)
print('bias(b) : ', '%.8f' % NL.intercept_)

X_test_poly = poly.fit_transform(X_test)
NL.score(X_test_poly, y_test) # 결정계수

X_test_poly = poly.fit_transform(X_test)
mean_squared_error(y_test, NL.predict(X_test_poly))

# Visualization
y_hat_test = NL.predict(X_test_poly)
plt.figure(figsize=(9, 6))
plt.plot(X_train, y_train, 'o', label='Train Data')
plt.plot(X_test, y_hat_test, 'r+', label='Predicated Value')
plt.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show

X_poly = poly.fit_transform(X)
y_hat2 = NL.predict(X_poly)

plt.figure(figsize=(9, 6))
ax1 = sns.distplot(y, hist=False, label='y')
ax2 = sns.distplot(y_hat2, hist=False, label='y_hat', ax=ax1)
plt.ylim(0, 0.07)
plt.show()

DF3 = DF[['mpg', 'cylinders', 'displacement', 'weight']]
DF3.head(3)

X = DF3[['displacement', 'weight']]
y = DF3['mpg']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=2045
                                                    )

print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)


MR = LinearRegression()
MR.fit(X_train, y_train)

print('weight(w) : ', MR.coef_)
print('bias(b) : ', '%.8f' % MR.intercept_)

MR.score(X_test, y_test) # R-Square

mean_squared_error(y_test, MR.predict(X_test))

y_hat3 = MR.predict(X_test)

plt.figure(figsize = (9, 6))
ax1 = sns.distplot(y_test, hist=False, label='y_test')
ax2 = sns.distplot(y_hat3, hist=False, label='y_hat', ax=ax1)
plt.ylim(0, 0.07)
plt.show()

# final visualization

y_hat3 = MR.predict(X_test)

plt.figure(figsize=(9, 6))
ax1 = sns.distplot(y_test, hist=False, label='y_test')
ax2 = sns.distplot(y_hat1, hist=False, label='y_hat', ax=ax1)
ax3 = sns.distplot(y_hat2, hist=False, label='y_hat', ax=ax1)
ax4 = sns.distplot(y_hat3, hist=False, label='y_hat', ax=ax1)
plt.ylim(0, 0.07)
plt.show()







