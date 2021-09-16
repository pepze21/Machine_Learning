import seaborn as sns # Visualizing distributions of data
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


DF = sns.load_dataset('mpg')
X = DF[['weight']]
y = DF['mpg']

fig = plt.figure(figsize=(9, 6))
sns.regplot(x=X, y=y)
plt.xlabel('weight_Without_Scaling')
plt.show()

# Normalization (in [0, 1])
scaler1 = MinMaxScaler()
X_Norm = scaler1.fit_transform(X)
fig = plt.figure(figsize = (9, 6))
sns.regplot(x = X_Norm, y=y)
plt.xlabel('weight_With_Normalization')
plt.show()

# Standardization (X_Stan ~ N(0,1))
scaler2 = StandardScaler()
X_Stan = scaler2.fit_transform(X)
fig = plt.figure(figsize = (9, 6))
sns.regplot(x=X_Stan, y=y)
plt.xlabel('weigh_With_Standardization')
plt.show()


