import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', 50)


data = pd.read_csv('data/sp500.csv', parse_dates=['Date'])
data.info()

# # double Y
# fig, ax_close = plt.subplots(figsize=(15, 6))
# ax_volume = ax_close.twinx()
# ax_close.plot(data['Date'], data['Close'], color='royalblue', lw=2, label='Close')
# ax_volume.plot(data['Date'], data['Volume'], color='tomato', lw=1, ls='dashed', label='Volume')
# handles_close, labels_close = ax_close.get_legend_handles_labels()
# handles_volume, labels_volume = ax_volume.get_legend_handles_labels()
# ax_close.legend(handles_close + handles_volume, labels_close + labels_volume)
#


from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import optuna
from functools import partial

data_prophet = data.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]
data_train, data_test = train_test_split(data_prophet, test_size=0.1, random_state=23,
                                         shuffle=False)


def cross_validate(data, n_splits=3, params=None):
    params = {} if params is None else params
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for index_train, index_valid in tscv.split(data):
        model = init_model(params)
        data_train, data_valid = data.loc[index_train], data.loc[index_valid]
        with suppress_stdout_stderr():
            model.fit(data_train)
        data_pred = model.predict(data_valid.drop('y', axis=1))

        loss = mean_squared_error(y_true=data_valid['y'], y_pred=data_pred['yhat'], squared=False)
        losses.append(loss)
    loss_mean, loss_std = np.mean(losses), np.std(losses)
    print(f'Ave loss = {loss_mean:.4e} +- {loss_std:.4e}')
    return loss_mean


def suggest_params(trial):
    params = {
        'changepoint_prior_scale': trial.suggest_loguniform('changepoint_prior_scale', 0.001, 0.5),
        'seasonality_prior_scale': trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10),
        'holidays_prior_scale': trial.suggest_loguniform('holidays_prior_scale', 0.01, 10),
        'seasonality_mode': trial.suggest_categorical(
            'seasonality_mode', ['additive', 'multiplicative']
        ),
        'changepoint_range': trial.suggest_uniform('changepoint_range', 0.8, 0.95),
        'yearly_seasonality': trial.suggest_int('yearly_seasonality', 10, 15),
        'weekly_seasonality': trial.suggest_int('weekly_seasonality', 3, 6)
    }
    return params


def init_model(params: dict):
    model = Prophet(growth='linear', daily_seasonality=False, **params)
    model.add_country_holidays(country_name='US')
    return model


def objective(trial, data, n_splits):
    params = suggest_params(trial=trial)
    loss = cross_validate(data, n_splits=n_splits, params=params)
    return loss

import os

class suppress_stdout_stderr(object):
    """
    A context manager for doing a 'deep suppression' of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience
        self._opt_loss = np.inf  # the smaller, the better!!!
        self._consecutive_no_improve_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if study.best_trial.value < self._opt_loss:
            self._opt_loss = study.best_trial.value
            self._consecutive_no_improve_count = 0
        else:
            self._consecutive_no_improve_count += 1

        if self._consecutive_no_improve_count >= self.patience:
            print(f'There is no improvement after {self.patience} trials. '
                  f'Stop the parameter optimization with opt loss = {self._opt_loss:.4e}.')
            study.stop()



early_stopping = EarlyStoppingCallback(patience=10)
sampler = optuna.samplers.TPESampler(seed=23)
study = optuna.create_study(direction='minimize', sampler=sampler, study_name='min_rmse')
study.optimize(partial(objective, data=data_train, n_splits=3), n_trials=100, callbacks=[early_stopping])
best_params = study.best_trial.params
best_loss = study.best_trial.value
