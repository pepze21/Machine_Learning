# 15:03

# import warnings
# warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

DF = pd.read_csv('https://raw.githubusercontent.com/rusita-ai/pyData/master/Default.csv')
DF.info()
DF.head()

# EDA
DF.default.value_counts()


plt.figure(figsize=(9, 6))
plt.boxplot([DF[DF.default == 'No'].balance,
            DF[DF.default == 'Yes'].balance
            ],
            labels = ['No', 'Yes']
           )
plt.show()

### Q. scaler를 여러개 합성해서 써도 되나? 같은 문법으로?
## Data Preprocessing
# Standardization
X = DF[['balance']]
y = DF['default']

scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)

X_Scaled[:5]



X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y,
                                                    test_size=0.3,
                                                    random_state = 0)
print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)


## Modeling

from sklearn.linear_model import LogisticRegression
Model_lr = LogisticRegression()
Model_lr.fit(X_train, y_train)

y_hat = Model_lr.predict(X_test)
y_hat

# training accuracy
Model_lr.score(X_train, y_train)
# test accuracy
Model_lr.score(X_test, y_test)

# 'default' == 'No'(상환) 기준
confusion_matrix(y_test, y_hat)

# 'default' == 'yes'(연체) 기준
confusion_matrix(y_test, y_hat, labels = ['Yes', 'No'])

# No(상환) 기준 << 무쓸모
print(accuracy_score(y_test, y_hat))
print(precision_score(y_test, y_hat, pos_label='No'))
print(recall_score(y_test, y_hat, pos_label = 'No'))

# Yes(연체) 기준 << 이게 의미있는 것
print(accuracy_score(y_test, y_hat))
print(precision_score(y_test, y_hat, pos_label='Yes'))
print(recall_score(y_test, y_hat, pos_label = 'Yes'))


f1_score(y_test, y_hat, pos_label='No')
f1_score(y_test, y_hat, pos_label='Yes')


print(classification_report(y_test, y_hat,
                            target_names=['No', 'Yes'],
                            digits=5))