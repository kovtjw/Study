from sklearn.datasets import load_iris, load_wine, load_breast_cancer,load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
iris = load_iris
wine = load_wine()
cancer = load_breast_cancer()
base_model= make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

bagging_model = BaggingClassifier(base_model,n_estimators=10, max_samples=0.5, max_features= 0.5)

cross_val = cross_validate(
    estimator= base_model,
    X = iris.data, y=iris.target,
    cv = 5
)
print('avg fit time:{} (+/-{})'.format(cross_val['fit_time'].mean(), cross_val['fit_time'].std()))
print('avg score time:{} (+/-{})'.format(cross_val['score time'].mean(), cross_val['score time'].std()))
print('avg test score:{} (+/-{})'.format(cross_val['test score'].mean(), cross_val['test score'].std()))

      