import numpy as np
import pandas as pd


import lightgbm as lgb
from lightgbm import LGBMClassifier

from  bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score


import warnings

warnings.filterwarnings('ignore')

SEED = 42

np.random.seed(SEED)
bounds_LGB = {
    'num_leaves': (100, 800), 
    'min_data_in_leaf': (0, 150),
    'bagging_fraction' : (0.3, 0.9),
    'feature_fraction' : (0.3, 0.9),
    'min_child_weight': (0.01, 1.),   
    'reg_alpha': (0.01, 1.), 
    'reg_lambda': (0.01, 1),
    'max_depth':(6, 23),
}

def build_lgb(x, y, val_x, val_y, init_points=30, n_iter=50, cv=4, param=True, verbose=2, is_test=False, SEED=42):
    def LGB_bayesian(
        num_leaves, 
        bagging_fraction,
        feature_fraction,
        min_child_weight, 
        min_data_in_leaf,
        max_depth,
        reg_alpha,
        reg_lambda,
         ):
        # LightGBM expects next three parameters need to be integer. 
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)

        assert type(num_leaves) == int
        assert type(min_data_in_leaf) == int
        assert type(max_depth) == int

        params = {
                  'num_leaves': num_leaves, 
                  'min_data_in_leaf': min_data_in_leaf,
                  'min_child_weight': min_child_weight,
                  'bagging_fraction' : bagging_fraction,
                  'feature_fraction' : feature_fraction,
                  'learning_rate' : 0.05,
                  'max_depth': max_depth,
                  'reg_alpha': reg_alpha,
                  'reg_lambda': reg_lambda,
                  'objective': 'binary',
                  'save_binary': True,
                  'seed': SEED,
                  'feature_fraction_seed': SEED,
                  'bagging_seed': SEED,
                  'drop_seed': SEED,
                  'data_random_seed': SEED,
                  'boosting': 'gbdt', 
                  'verbose': 0,
                  'boost_from_average': True,
                  'metric':'auc',
                  'n_estimators': 1000,
                  'n_jobs': -1,
        }    

        ## set reg options
        model = lgb.LGBMClassifier(**params)
        model.fit(x, y, eval_set=(val_x, val_y), early_stopping_rounds=30, verbose=0)
        pred = model.predict(val_x)
        score = f1_score(val_y, pred)
        return score
    
    optimizer = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=SEED, verbose=verbose)
    init_points = init_points
    n_iter = n_iter

    optimizer.maximize(init_points=init_points, n_iter=n_iter)
    
    param_lgb = {
        'min_data_in_leaf': int(optimizer.max['params']['min_data_in_leaf']), 
        'num_leaves': int(optimizer.max['params']['num_leaves']), 
        'learning_rate': 0.05,
        'min_child_weight': optimizer.max['params']['min_child_weight'],
        'bagging_fraction': optimizer.max['params']['bagging_fraction'], 
        'feature_fraction': optimizer.max['params']['feature_fraction'],
        'reg_lambda': optimizer.max['params']['reg_lambda'],
        'reg_alpha': optimizer.max['params']['reg_alpha'],
        'max_depth': int(optimizer.max['params']['max_depth']), 
        'objective': 'binary',
        'save_binary': True,
        'seed': SEED,
        'feature_fraction_seed': SEED,
        'bagging_seed': SEED,
        'drop_seed': SEED,
        'data_random_seed': SEED,
        'boosting': 'gbdt', 
        'verbose': -1,
        'boost_from_average': True,
        'metric':'auc',
        'n_estimators': 1000,
        'n_jobs': -1,
    }

    params = param_lgb.copy()

    model = lgb.LGBMClassifier(**params)
    model.fit(x, y, eval_set=(val_x, val_y), early_stopping_rounds=30, verbose=0)
    if param:
        return model, params
    else:
        return model
    
def preprocessing(_df):
    df = _df.copy()
    # write what you want
    return df
datapath = '../_data/dacon/cardiovascular disease/'
train = pd.read_csv(datapath + 'train.csv', index_col='id')
test = pd.read_csv(datapath + 'test.csv', index_col='id')

sub = pd.read_csv(datapath + 'sample_submission.csv', index_col='id')

tr_X = preprocessing(train.drop('target', axis=1))
test_X = preprocessing(test)

tr_y = train['target']

from sklearn.model_selection import StratifiedKFold

n_fold = 10
sf = StratifiedKFold(n_fold, shuffle=True, random_state=SEED)

preds = []
thresholds = []
c = 1
for tr_idx, val_idx in sf.split(tr_X, tr_y):
    print('#'*25, f'CV {c}')
    model, _ = build_lgb(tr_X.iloc[tr_idx], tr_y.iloc[tr_idx], tr_X.iloc[val_idx], tr_y.iloc[val_idx], 15, 15)
    _preds = model.predict(test_X)
    preds.append(_preds)
    c += 1
    
sub['target'] = np.where(np.mean(preds, 0)>0.5, 1, 0)
sub.to_csv(datapath+"hi2.csv", index = False)