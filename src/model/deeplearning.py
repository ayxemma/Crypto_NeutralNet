from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.backend import clear_session

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import gc
import pickle
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime


def lstm_prediction():
    """
    start the burn in with a year's data - 9 month and 3 month split
    retrain model every 10 days
    """
    #    df_ftr = pd.read_csv(r'../data/ftr_tgt.csv')
    df_ftr = read_cleaned_data()
    lookback = 60
    target = 'target'
    df_ftr = df_ftr.dropna()
    # df_ftr['date'] = pd.to_datetime(df_ftr['date'])
    # df_ftr = df_ftr.set_index('date')
    # df_ftr = df_ftr.resample('3Min').last()
    # df_ftr[target] = df_ftr[target]+1
    df_burn = df_ftr.loc[df_ftr.index.year == 2018]
    train_period = 9
    test_period = 11
    n_period = 12
    df_train = df_burn.loc[df_burn.index.month <= train_period]
    df_val = df_burn.loc[((df_burn.index.month > train_period) & (df_burn.index.month <= test_period))]
    df_test = df_burn.loc[((df_burn.index.month > test_period) & (df_burn.index.month <= n_period))]

    ftr_cols = [x for x in df_train.columns if x != 'target']
    print(f'train data shape is {df_train.shape}')
    print(f'validation data shape is {df_val.shape}')

    mm_scaler = preprocessing.MinMaxScaler()
    X_train = mm_scaler.fit_transform(df_train[ftr_cols])
    df_check = pd.DataFrame(X_train)
    X_val = mm_scaler.transform(df_val[ftr_cols])
    X_test = mm_scaler.transform(df_test[ftr_cols])
    df_check = pd.DataFrame(X_val)
    X_train_lstm = []
    y_train_lstm = []
    for i in range(lookback, len(X_train), 3):
        X_train_lstm.append(X_train[i - lookback:i, :])
        y_train_lstm.append(df_train[target].iloc[i])

    X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
    X_val = np.concatenate((X_train, X_val), axis=0)
    X_test = np.concatenate((X_val, X_test), axis=0)
    X_val_lstm = []
    y_val_lstm = []
    X_test_lstm = []
    y_test_lstm = []
    for i in range(len(X_train), len(X_val), 3):
        X_val_lstm.append(X_val[i - lookback:i, :])
        y_val_lstm.append(df_val[target].iloc[i - len(X_train)])

    for i in range(len(X_val), len(X_test), 3):
        X_test_lstm.append(X_test[i - lookback:i, :])
        y_test_lstm.append(df_test[target].iloc[i - len(X_test)])

    X_val_lstm, y_val_lstm = np.array(X_val_lstm), np.array(y_val_lstm)
    X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)

    mdl_nm = 'LSTM'
    model = create_lstm_model(n_lookback=X_train_lstm.shape[1], n_features=X_train_lstm.shape[2], lr=0.001)
    earlyStopping = EarlyStopping(monitor='val_accuracy', patience=300, verbose=0, mode='max', min_delta=0.001)
    mcp_save = ModelCheckpoint(f'../models/{mdl_nm}_train_split{train_period}vs{n_period - train_period}.h5',
                               save_best_only=True, monitor='val_accuracy', mode='max')
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
    csv_logger = CSVLogger(f'../models/{mdl_nm}_train_split{train_period}vs{n_period - train_period}.log')

    callback_ls = [mcp_save, csv_logger, mcp_save]
    class_weight = {0: 20, 1: 80}
    print('start run model')
    history = model.fit(X_train_lstm,  # Features
                        y_train_lstm,  # Target
                        epochs=2000,  # Number of epochs
                        verbose=1,  # No output
                        batch_size=2 ** 10,  # Number of observations per batch
                        validation_data=(X_val_lstm, y_val_lstm),
                        callbacks=callback_ls,
                        class_weight=class_weight)  # Data for evaluation
    plot_model_loss(history, outfile=f'../models/{mdl_nm}_split{train_period}vs{n_period - train_period}')
    model = load_model(f'../models/{mdl_nm}_train_split{train_period}vs{n_period - train_period}.h5')
    train_pred_y = model.predict(X_train_lstm)
    # test_pred_y = model.predict(X_test_lstm)
    val_pred_y = model.predict(X_val_lstm)
    test_pred_y = model.predict(X_test_lstm)
    #
    #
    df_train_pred = pd.DataFrame(y_train_lstm)
    df_train_pred.columns = ['target']
    df_train_pred['pred_prob'] = train_pred_y
    plot_roc_curve(df_train_pred, target, 'train')
    prob_cut = Find_Optimal_Cutoff(df_train_pred[target], df_train_pred['pred_prob'])
    df_train_pred['pred_bi'] = (df_train_pred['pred_prob'] > prob_cut[0])
    confusion_matrix(df_train_pred[target], df_train_pred['pred_bi'])
    print(classification_report(df_train_pred[target], df_train_pred['pred_bi']))

    df_val_pred = pd.DataFrame(y_val_lstm)
    df_val_pred.columns = ['target']
    df_val_pred['pred_prob'] = val_pred_y
    plot_roc_curve(df_val_pred, target, 'val')
    prob_cut = Find_Optimal_Cutoff(df_val_pred[target], df_val_pred['pred_prob'])
    df_val_pred['pred_bi'] = (df_val_pred['pred_prob'] > prob_cut[0])
    confusion_matrix(df_val_pred[target], df_val_pred['pred_bi'])
    print(classification_report(df_val_pred[target], df_val_pred['pred_bi']))

    df_test_pred = pd.DataFrame(y_test_lstm)
    df_test_pred.columns = ['target']
    df_test_pred['pred_prob'] = test_pred_y
    plot_roc_curve(df_test_pred, target, 'test')
    prob_cut = Find_Optimal_Cutoff(df_test_pred[target], df_test_pred['pred_prob'])
    df_test_pred['pred_bi'] = (df_test_pred['pred_prob'] > prob_cut[0])
    confusion_matrix(df_test_pred[target], df_test_pred['pred_bi'])
    print(classification_report(df_test_pred[target], df_test_pred['pred_bi']))
    # df_val_pred = df_val_pred.reset_index(drop=True)
    # plt.figure()
    # plt.scatter(df_val_pred.index, df_val_pred[target], s=5, marker='.')
    # plt.savefig(cfg.local_projfolder + 'temp/visualization/plot.png')
    clear_session()
    del model
    gc.collect()


def read_model_log(mdl_nm, train_period, n_period):
    log = pd.read_csv(f'../models/{mdl_nm}_train_split{train_period}vs{n_period - train_period}.log')


def create_lstm_model(n_lookback, n_features, lr, init='glorot_normal'):
    dropout_pct = 0.40
    l1l2_reg = 0.00002
    dtype = 'float32'
    _input1 = Input(shape=[n_lookback, n_features])
    input_dropout = SpatialDropout1D(dropout_pct)(_input1)
    lstm_out = LSTM(32, kernel_initializer=init, return_sequences=True, dtype=dtype)(input_dropout)
    lstm_out = SpatialDropout1D(dropout_pct, dtype=dtype)(lstm_out)
    lstm_out = LSTM(16, kernel_initializer=init, return_sequences=False, dtype=dtype)(lstm_out)
    lstm_out = Dropout(dropout_pct, dtype=dtype)(lstm_out)
    lstm_out = LeakyReLU(dtype=dtype)(lstm_out)
    final_output = Dense(8, kernel_initializer=init, kernel_regularizer=regularizers.l1_l2(l1l2_reg, l1l2_reg),
                         dtype=dtype)(lstm_out)
    final_output = Dropout(dropout_pct)(final_output)
    final_output = LeakyReLU(dtype=dtype)(final_output)
    final_output = Dense(4, kernel_initializer=init, kernel_regularizer=regularizers.l1_l2(l1l2_reg, l1l2_reg),
                         dtype=dtype)(final_output)
    final_output = Dropout(dropout_pct)(final_output)
    final_output = LeakyReLU(dtype=dtype)(final_output)
    final_output = Dense(1, activation='sigmoid', kernel_initializer=init,
                         kernel_regularizer=regularizers.l1_l2(l1l2_reg, l1l2_reg), dtype=dtype)(final_output)
    model = Model(inputs=_input1, outputs=final_output)
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def load_log(mdl_nm, train_period, n_period):
    log = pd.read_csv(f'../models/{mdl_nm}_train_split{train_period}vs{n_period - train_period}.log')
    log[['accuracy', 'val_accuracy']].plot()
    plt.savefig(f'../visualization/{mdl_nm}_train_split{train_period}vs{n_period - train_period}_accuracy_logview.png')
    log[['loss', 'val_loss']].plot()
    plt.savefig(f'../visualization/{mdl_nm}_train_split{train_period}vs{n_period - train_period}_loss_logview.png')
