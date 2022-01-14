from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

iris = load_iris()

print(len(iris.data))   # 150
print(iris.keys())      # dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])


# K-fold
dt = DecisionTreeClassifier()
kfold = KFold(n_splits=20, shuffle = True)   # n_split : 데이터 분할 수 // shuffle : 매번 데이터를 분할하기 전 섞을지 말지 여부

score = cross_val_score(dt, iris.data,iris.target, cv=kfold, scoring='accuracy')
                       #모델, feature, target, cv(분할 설정 값), scoring(평가방법)
print('k-fold 교차검증 결과의 평균 : ', score.mean())
 
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))