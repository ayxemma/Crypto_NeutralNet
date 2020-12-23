#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:59:18 2019

@author: ayx
"""

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

import pandas as pd
import talib as ta
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import gc
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
from functools import reduce
import src.model.deeplearning as NN

import os

# os.chdir(r'/Users/ayx/Documents/Trading/CryptoCoin/Okex/deeplearning/')
import src.utils as utils


# sys.path.append(r'/Users/ayx/Documents/Trading/CryptoCoin/Okex/deeplearning/src')
# os.chdir(r'/Users/ayx/Documents/Trading/CryptoCoin/Okex/deeplearning/src')


# os.chdir(r'/home/wintersunrise11_gmail_com/deeplearning/src')

# os.chdir(r'/home/cloud-user/projects/predicting_financials/src/temp/src')
#
# tf.compat.v1.disable_eager_execution()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)

def read_data():
    df = pd.read_pickle(r'./data/eth_usdt_all.pkl')
    # df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.loc[df.index > pd.to_datetime('2018-1-1')]
    return df


def build_features_resample():
    """
    resembles the real trading strategy to avoid overlapping by using rolling
    :return:
    """
    df = read_data()
    ls_interval = ['30Min', '60Min']
    ls_N = [30, 60]
    dic_df = {}
    minute_interval = 2880
    prc_col = ['open', 'high', 'low', 'close']
    vol_col = ['volume']
    df_ftr = df[['open', 'high', 'low', 'close', 'volume']]

    ls_ftr = []
    for t in tqdm(range(2880, len(df_ftr), 10)):
        df_ftr_t = df_ftr.iloc[t - 2880:t]
        dic_ftr = {}
        dic_df = {}
        dic_df['date'] = df_ftr.index[100]
        dic_ftr['date'] = df_ftr.index[100]

        # construct different frequency timeseris OHLCV for each minute
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            # resample into time frequency
            df_ts = df_ftr_t['close'].resample(ts_interval).last().to_frame()
            df_ts['volume'] = df_ftr_t['volume'].resample(ts_interval).sum()
            df_ts['close'] = df_ts['close'].fillna(method='ffill')
            df_ts['volume'] = df_ts['volume'].fillna(0)
            dic_df[ts_interval] = df_ts.copy()

        # volume break out and volume std
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            df_ts = dic_df[ts_interval]
            vol_std = df_ts['volume'].iloc[:-1].std()
            vol_median = df_ts['volume'].iloc[:-1].median()
            vol_bo = df_ts['volume'].iloc[-1] > (vol_median + 2 * vol_std)
            dic_ftr.update({f'vol_bo_{ts_interval}': vol_bo})
            dic_ftr.update({f'vol_std_{ts_interval}': vol_std})

        # price break out
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            df_ts = dic_df[ts_interval]
            upper, middle, lower = ta.BBANDS(df_ts['close'], matype=ta.MA_Type.T3)
            up_bo = df_ts['close'].iloc[-1] > upper[-1]
            low_bo = df_ts['close'].iloc[-1] < lower[-1]
            dic_ftr.update({f'up_bo_{ts_interval}': up_bo})
            dic_ftr.update({f'low_bo_{ts_interval}': low_bo})

        # price volatility
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            df_ts = dic_df[ts_interval]
            price_std = df_ts['close'].std()
            dic_ftr.update({f'price_std_{ts_interval}': price_std})

        # price pct_change
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            df_ts = dic_df[ts_interval]
            dic_ftr.update({f'price_pctchg_{ts_interval}': df_ts['close'].iloc[-1] / df_ts['close'].iloc[-2] - 1})

        # price slope
        for i in range(len(ls_interval)):
            ts_interval = ls_interval[i]
            df_ts = dic_df[ts_interval]
            dic_ftr.update({f'price_slope_{ts_interval}':
                                ta.LINEARREG_SLOPE(df_ts['close'].iloc[-3:], timeperiod=3)[-1]})

        ls_ftr.append(dic_ftr)
        ## MAs
    #        for i in range(len(ls_interval)):
    #            ts_interval = ls_interval[i]
    #            df_ts = dic_df[ts_interval]
    #            dic_ftr.update({f'MA{ts_interval}':
    #                df_ts['close'].iloc[-3:].mean()})

    df_ftr_cal = pd.DataFrame(ls_ftr)
    df_ftr_cal.to_pickle(r'../data/df_ftr_cal.pkl')
    df_ftr_cal['date'] = df_ftr.index[2880:]
    df_ftr_cal = df_ftr_cal.set_index('date')
    df_ftr_cal.to_pickle(r'../data/df_ftr_cal.pkl')


def target_calculation_MaxMinReturn():
    df = read_data()
    ls_interval = ['30Min', '60Min']
    ls_N = [30, 60]
    df_ftr = df[['open', 'high', 'low', 'close', 'volume']]
    # price max increase, max decline, increase><decline
    df_ftr['max_increase'] = df['close'].rolling(minute_interval).apply(lambda x: x[-1] / min(x[:-1]))
    df_ftr['max_decline'] = df['close'].rolling(minute_interval).apply(lambda x: x[-1] / max(x[:-1]))
    df_ftr['up_gt_down'] = abs(df_ftr['max_increase']) > abs(df_ftr['max_decline'])
    df_ftr['down_gt_up'] = abs(df_ftr['max_decline']) > abs(df_ftr['max_increase'])
    df_ftr = df_ftr.set_index('date')

    df_ftr_cal = df_ftr_cal.set_index('date')
    df_ftr_cal['max_increase'] = df_ftr['max_increase']
    df_ftr_cal['max_decline'] = df_ftr['max_decline']
    df_ftr_cal['up_gt_down'] = df_ftr['up_gt_down']
    df_ftr_cal['down_gt_up'] = df_ftr['down_gt_up']

    ## convert true false values
    df_ftr_cal = df_ftr_cal * 1
    df_ftr_cal.to_pickle(r'../data/df_ftr_cal_all_features.pkl')

    ## forward n hour return
    fwd_t = 2880
    df_rtn = pd.DataFrame()
    df_rtn['max_up_return'] = df['close'].rolling(fwd_t).max().shift(-(fwd_t - 1)) / df['close'] - 1
    df_rtn['max_down_return'] = df['close'].rolling(fwd_t).min().shift(-(fwd_t - 1)) / df['close'] - 1
    df_rtn['fwd_return'] = df_rtn['max_up_return']
    idx = abs(df_rtn['max_up_return']) < abs(df_rtn['max_down_return'])
    df_rtn.loc[idx, 'fwd_return'] = df_rtn.loc[idx, 'max_down_return']

    df_ftr_cal['fwd_return'] = df_rtn['fwd_return']
    df_ftr_cal.to_pickle(r'../data/df_ftr_cal_all_features_fwdreturn.pkl')


def build_features_rolling(df):
    ## script
    ls_interval = ['1Min', '5Min', '15Min']
    ls_N = [5, 15, 30]
    dic_df = {}
    minute_interval = 2880
    prc_col = ['open', 'high', 'low', 'close']
    vol_col = ['volume']
    df_ftr = df[['open', 'high', 'low', 'close', 'volume']]
    for i in range(len(ls_interval)):
        ts_interval = ls_interval[i]
        df_ts = df['close'].to_frame()
        df_ts['open'] = df['open'].shift(ls_N[i] - 1)
        df_ts['high'] = df['high'].rolling(ls_N[i]).max()
        df_ts['low'] = df['low'].rolling(ls_N[i]).min()
        df_ts['volume'] = df['volume'].rolling(ls_N[i]).sum()
        dic_df[ts_interval] = df_ts

    # volume break out and volume std
    for i in range(len(ls_interval)):
        ts_interval = ls_interval[i]
        df_ts = dic_df[ts_interval]
        df_ts['vol_std'] = df_ts['volume'].rolling(minute_interval).std()
        df_ts['vol_median'] = df_ts['volume'].rolling(minute_interval).median()
        df_ts['vol_bo'] = df_ts['volume'] > (df_ts['vol_median'] + 4 * df_ts['vol_std'])
        df_ftr[f'vol_bo_{ts_interval}'] = df_ts['vol_bo']
        df_ftr[f'vol_std_{ts_interval}'] = df_ts['vol_std']

    # price break out
    for i in range(len(ls_interval)):
        ts_interval = ls_interval[i]
        df_ts = dic_df[ts_interval]
        upper, middle, lower = ta.BBANDS(df_ts['close'], matype=ta.MA_Type.T3)
        df_ts['up_bo'] = df_ts['close'] > upper
        df_ts['low_bo'] = df_ts['close'] < lower
        df_ftr[f'up_bo_{ts_interval}'] = df_ts['up_bo']
        df_ftr[f'low_bo_{ts_interval}'] = df_ts['low_bo']

    # price volatility
    df_ftr['price_std'] = df['close'].rolling(minute_interval).std()

    # price pct_change
    for i in range(len(ls_interval)):
        ts_interval = ls_interval[i]
        df_ftr[f'price_pctchg_{ts_interval}'] = df['close'].rolling(ls_N[i] + 1).apply(lambda x: x[-1] / x[0] - 1)

    # price max increase, max decline, increase><decline
    df_ftr['max_increase'] = df['close'].rolling(minute_interval).apply(lambda x: x[-1] / min(x[:-1]))
    df_ftr['max_decline'] = df['close'].rolling(minute_interval).apply(lambda x: x[-1] / max(x[:-1]))
    df_ftr['up_gt_down'] = abs(df_ftr['max_increase']) > abs(df_ftr['max_decline'])
    df_ftr['down_gt_up'] = abs(df_ftr['max_decline']) > abs(df_ftr['max_increase'])

    # price slope
    for i in range(len(ls_interval)):
        ts_interval = ls_N[i]
        df_ftr[f'price_pctchg_{ts_interval}'] = df['close'].rolling(ls_N[i] * 2 + 1).apply(
            lambda x: ta.LINEARREG_SLOPE(np.asarray([x[0], x[ts_interval], x[-1]], dtype='f8'), timeperiod=3)[-1])

    ## MAs
    for i in range(len(ls_interval)):
        ts_interval = ls_interval[i]
        df_ftr[f'MA{ts_interval}'] = df_ftr['close'].rolling(ls_N[i]).mean()

    ## convert true false values
    df_ftr = df_ftr * 1
    df_ftr.to_csv(r'../data/ftr.csv')
    return df_ftr


# TODO: distribution analysis of feature and target return
# TODO: model forecast using volume based features, identify what happens after large volume

def get_target(df_ftr):
    ## modeling
    df_ftr = pd.read_csv(r'../data/ftr.csv')

    ## calculate return - in 12 hours
    fwd_pd = 720
    df_ftr['rolling_max_rtn'] = df_ftr['close'].rolling(fwd_pd).apply(lambda x: max(x[1:]) / x[0] - 1)
    df_ftr['rolling_min_rtn'] = df_ftr['close'].rolling(fwd_pd).apply(lambda x: min(x[1:]) / x[0] - 1)
    df_ftr['rolling_max_rtn'] = df_ftr['rolling_max_rtn'].shift(-(fwd_pd - 1))
    df_ftr['rolling_min_rtn'] = df_ftr['rolling_min_rtn'].shift(-(fwd_pd - 1))
    df_ftr['target'] = ((df_ftr['rolling_max_rtn'] > 0.024) & (df_ftr['rolling_min_rtn'] > -0.012)) * 1
    df_ftr = df_ftr.drop(['rolling_max_rtn', 'rolling_min_rtn'], axis=1)
    ## model
    df_ftr.to_pickle(r'../data/ftr_tgt.pkl')
    return df_ftr


def target_analysis(df_ftr, target):
    # cross time consistency
    df_ftr['date'] = pd.to_datetime(df_ftr['date'])
    df_ft
    r = df_ftr.set_index('date')
    df_ftr['year'] = df_ftr.index.year
    df_ftr['month'] = df_ftr.index.month
    df_tg
    t = df_ftr.groupby(['year', 'month'])[target].sum().reset_index()
    # df_tgt['date']=df_tgt.apply(lambda x: datetime.date(year=x['year'],month=x['month']))
    plt.figure()
    df_tgt[target].plot()
    plt.savefig(r'../visualization/target.png')

    # percentage of the target
    df_ftr[target].sum() / len(df_ftr)


def plot_model_loss(history, outfile):
    training_loss = history.history['loss']
    test_loss = history.history['val_loss']
    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    # Visualize loss history
    fig = plt.figure()
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(cfg.local_projfolder + f'../visualization/{outfile}_model_loss.png')


def read_cleaned_data():
    # df_ftr = pd.read_csv(r'../data/ftr_tgt.csv')
    # df_ftr.to_pickle(r'../data/ftr_tgt.pkl')
    df_ftr = pd.read_pickle(r'../data/ftr_tgt.pkl')
    return df_ftr


def get_timeseries_trading_data(df, ts_period):
    df_hrate = df['close'].resample(ts_period).max().to_frame()
    df_hrate.columns = ['high']
    df_lrate = df['close'].resample(ts_period).min().to_frame()
    df_lrate.columns = ['low']
    df_orate = df['close'].resample(ts_period).first().to_frame()
    df_orate.columns = ['open']
    df_crate = df['close'].resample(ts_period).last().to_frame()
    df_crate.columns = ['close']
    df_amount = df['volume'].resample(ts_period).sum().to_frame()
    df_amount.columns = ['volume']

    dfs = [df_orate, df_hrate, df_lrate, df_crate, df_amount]
    df_grp = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
    return df_grp


def hourly_technical_pred():
    pass


## for lstm model
if __name__ == '__main__':
    NN.lstm_prediction()
