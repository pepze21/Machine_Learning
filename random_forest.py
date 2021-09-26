import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from datetime import datetime


DF = sns.load_dataset('iris')

# print(DF.info())
# print(DF.head())
# DF.species.value_counts()

sns.pairplot(hue='species', data=DF)
plt.show()

# Data set
X = DF[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = DF['species']

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=2045
                                                    )
print('Train Data : ', X_train.shape, y_train.shape)
print('Test Data : ', X_test.shape, y_test.shape)

# RandomForestClassifier()
# random_state : 반복 실행시 동일한 결과 출력
# n_jobs=-1 : 모든 CPU 코어 사용
Model_rf = RandomForestClassifier(n_estimators=10,
                                  max_features=2,
                                  random_state=2045,
                                  n_jobs=-1
                                  )
# fit()                                  
Model_rf.fit(X_train, y_train)

# Colab CPU check
# !cat /proc/cpuinfo : 자세히 보기
# !cat /proc/cpuinfo | grep 'model name'
# cat : concatenate(파일 내용 출력)(리눅스 커맨드)
# proc : process
# >>> model name    : Intel(R) Xeon(R) CPU @ 2.20GHz
# windows key + pause/break == PC 정보

# predict()
y_hat = Model_rf.predict(X_test)

# Evaluate
print(confusion_matrix(y_test, y_hat))
print(accuracy_score(y_test, y_hat))

# **feature_Importance_**
# 어떤 변수가 얼만큼 중요하게 참조되었는지를 보여줌
# x_n과 y_hat의 관련도?
Model_rf.feature_importances_

plt.figure(figsize=(9, 6))
sns.barplot(Model_rf.feature_importances_,
            ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            )
plt.show()

# Hyperparameter Tuning
# n_estimators : num of Decision Tree
# max_features : 분할에 사용되는 Feature(x)의 개수 -> variable의 갯수를 말하는거임
# max_depth : max depth of Trees
# max_leaf_nodes : max num of leaf nodes
# min_samples_split : 분할을 위한 최소한의 샘플데이터 개수
# min_samples_leaf : 말단 노드가 되기 위한 최소한의 샘플데이터 개수
# 위 두개는 비슷한 용도로 보면 되는지?


# RandomForestClassifier()
Model_rf = RandomForestClassifier()

# GridSearch와 RandomSearch가 있는데 GridSearch를 더 많이 씀
# GridSearchCV Hyperparameters
params = {'n_estimators':[100, 300, 500, 700],
          'max_features':[1, 2, 3, 4],
          'max_depth':[1, 2, 3, 4, 5],
          'random_state':[2045]
          }

# GridSearchCV()

grid_cv = GridSearchCV(Model_rf,
                       param_grid=params,
                       scoring='accuracy', # grid에서 accuracy가 높은애를 찾는거임
                       cv = KFold(n_splits=5
                                  # , random_state=2045
                                  ), #cv를 사용하면 kfold들의 평균을 구해서 validation
                       refit=True,
                       n_jobs=-1
                       )

# fit()
# gridsearch는 params의 조합(case)가 많기 때문에, 실행시간이 좀 오래걸림
start_time = datetime.now()
grid_cv.fit(X_train, y_train)
end_time = datetime.now()
print('Elapsed Time : ', end_time - start_time)

# Best Accuracy
print(grid_cv.best_score_)

# Best Hyperparameter
print(grid_cv.best_params_)

# Best Model
Model_CV = grid_cv.best_estimator_

# Evaluation
y_hat = Model_CV.predict(X_test)
print(confusion_matrix(y_test, y_hat))
print(accuracy_score(y_test, y_hat))

