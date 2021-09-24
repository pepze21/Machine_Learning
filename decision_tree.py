# import warnings
# warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz # 설치 필요
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


DF = sns.load_dataset('iris')
# DF.info()
# print(DF.head())
# print(DF.species.value_counts())

sns.pairplot(hue='species', data=DF)
plt.show()

# data set
X = DF[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = DF['species']

# train_test_split()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=0)
print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier

Model_dt = DecisionTreeClassifier(random_state=0)
Model_dt.fit(X_train, y_train)

graphviz.Source(export_graphviz(Model_dt,
                                class_names = (['setosa', 'virginica', 'versicolor']),
                                feature_names = (['sepal_length', 'sepal_width', 'petal_length', 'petal_width']),
                                filled=True
                                )
                )

y_hat = Model_dt.predict(X_test)

# confusion matrix
confusion_matrix(y_test, y_hat)

print(accuracy_score(y_test, y_hat))
print(precision_score(y_test, y_hat, average = None))
print(recall_score(y_test, y_hat, average = None))

f1_score(y_test, y_hat, average = None)


# Model Pruning(가지치기)
Model_pr = DecisionTreeClassifier(max_depth=3,
                                  random_state=0)
Model_pr.fit(X_train, y_train)

graphviz.Source(export_graphviz(Model_pr,
                                class_names=(['setosa', 'virginica', 'versicolor']),
                                feature_names=(['sepal_length', 'sepal_width', 'petal_length', 'petal_width']),
                                filled=True
                                )
                )

# confusion matrix
y_hat = Model_pr.predict(X_test)
print(confusion_matrix(y_test, y_hat))

print(accuracy_score(y_test, y_hat))
print(precision_score(y_test, y_hat, average=None))
print(recall_score(y_test, y_hat, average=None))
f1_score(y_test, y_hat, average=None)


# **Feature Importance**
Model_pr.feature_importances_

plt.figure(figsize=(9,6))
sns.barplot(Model_pr.feature_importances_,
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
plt.show()

