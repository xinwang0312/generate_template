import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 50)


data = pd.read_csv('data/tabular_titanic.csv')
data.info()

feat_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
cat_cols = ['Sex', 'Embarked']

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=23)

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
cat_enc = pd.DataFrame(enc.fit_transform(data[cat_cols]), columns=enc.get_feature_names(cat_cols))
X = pd.concat([data[set(feat_cols) - set(cat_cols)], cat_enc], axis=1)
X['Age'].fillna(X['Age'].median(), inplace=True)

ridge = RidgeClassifier()

score_ridge = cross_val_score(ridge, X, data['Survived'], cv=cv, scoring='roc_auc')
print(f'score_ridge = {score_ridge.mean():.4f} +- {score_ridge.std():.4f}')
ridge.fit(X, data['Survived'])
fig, ax = plt.subplots()
sns.barplot(x=X.columns, y=ridge.coef_.ravel(), ax=ax)
fig.autofmt_xdate()

# lightgbm
data_lgb = data.copy()
le_dict = {}
for c in cat_cols:
    le = LabelEncoder()
    data_lgb[c] = le.fit_transform(data_lgb[c])
    le_dict[c] = le

lgb_clf = lgb.LGBMClassifier()

import warnings
warnings.simplefilter("ignore")

fit_params = {'categorical_feature': cat_cols}
score_lgb = cross_val_score(lgb_clf, data_lgb[feat_cols], data_lgb['Survived'], cv=cv, scoring='roc_auc',
                            fit_params=fit_params)
print(f'score_lgb = {score_lgb.mean():.4f} +- {score_lgb.std():.4f}')

X = lgb.Dataset(data=data_lgb[feat_cols], label=data_lgb['Survived'], categorical_feature=cat_cols, free_raw_data=False)
cv_results = lgb.cv(params=lgb_clf.get_params(), train_set=X, folds=cv, verbose_eval=-1, metrics='auc')
print(f"score_lgb = {cv_results['auc-mean'][-1]:.4f} +- {cv_results['auc-stdv'][-1]:.4f}")



dtrain = lgb.Dataset(data=data_lgb[feat_cols], label=data_lgb['Survived'],
                     categorical_feature=cat_cols, free_raw_data=False)

import optuna

param_fixed = {
    'objective': 'binary',
    'metric': 'auc',  # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    'verbosity': -1,
    'boosting_type': 'gbdt'
}


def objective(trial):
    param_opt = {
        'max_depth': trial.suggest_int('bagging_freq', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256)
    }
    param = {**param_fixed, **param_opt}
    cv_results = lgb.cv(params=param, train_set=dtrain, folds=cv)
    return cv_results['auc-mean'][-1]


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print('Number of finished trials: {}'.format(len(study.trials)))

print('Best trial:')
trial = study.best_trial

print('  Value: {}'.format(trial.value))

print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))


clf_lgb = lgb.train(params={**param_fixed, **trial.params}, train_set=dtrain)
feat_imp_split = clf_lgb.feature_importance(importance_type='split')
feat_imp_gain = clf_lgb.feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': clf_lgb.feature_name(), 'gain': feat_imp_gain, 
                         'split': feat_imp_split})

fig_width = np.ceil(len(feat_imp) * 0.8).astype(int)
x = np.arange(len(feat_imp))
width = 0.35  # the width of the bars

fig, ax_gain = plt.subplots(figsize=(fig_width, 6))
ax_split = ax_gain.twinx()
legend, labels, axes = [], ['gain', 'split'], [ax_gain, ax_split]
for ax, l, w, c in zip(axes, labels, [-width, width], ['royalblue', 'tomato']):
    rects = ax.bar(x, feat_imp[l], width=w, align='edge', label=l, color=c, alpha=0.7)
    legend.append(rects)
    ax.set_ylabel(l)
ax_gain.legend(legend, labels)
ax_gain.set_xticks(x)
# ax_gain.set_xticklabels(feat_imp['name'])
fig.autofmt_xdate()