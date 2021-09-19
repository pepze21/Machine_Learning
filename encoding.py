import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

DF = sns.load_dataset('mpg')

type(DF.origin[0]) #str

DF.origin.value_counts() # typeusa 249 japan 79 europe 70

X = DF[['origin']]

# Integer encoding (== Lable encoding)
encoder1 = LabelEncoder()
LE = encoder1.fit_transform(X) # array([2 2 2 ... ])

print(X)
print(LE) # 0 : europe, 1 : japan, 2 : usa

# OneHotEncoder
encoder2 = OneHotEncoder()
OHE = encoder2.fit_transform(X)

print(OHE) # (index, first location of 1)   
print(OHE.toarray())