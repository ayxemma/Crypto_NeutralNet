

class BlockingTimeSeriesSplit():
    """
    method take from -
    https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start


def model_cross_validation():
    """
    use randomforest to predict

    """
    df_ftr = pd.read_pickle(r'./data/df_ftr_cal_all_features_fwdreturn.pkl')
    df_ftr = df_ftr.dropna()
    df_ftr = df_ftr * 1
    df_ftr = df_ftr.loc[df_ftr.index < pd.to_datetime('2019-8-1')]
    target = 'fwd_return'
    ftr_cols = [x for x in df_ftr.columns if x != target]

    param_grid = {'n_estimators': [400, 600, 100],
                  'max_depth': [5, 10],
                  'max_features': [5, 10],
                  'min_samples_split': [10, 20],
                  'min_samples_leaf': [10, 20],
                  'max_samples': [0.1, 0.25]}

    param_grid = {'n_estimators': [10, 30],
                  'max_depth': [5, 3],
                  'max_features': [5, 3],
                  'min_samples_split': [10, 3],
                  'min_samples_leaf': [10, 3],
                  'max_samples': [0.1, 0.1]}

    estimator = RandomForestRegressor()
    btscv = BlockingTimeSeriesSplit(n_splits=5)
    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(estimator, param_grid, scoring=make_scorer(r2_score), n_jobs=-1,
                               iid='deprecated', refit=True, cv=tscv, verbose=0)
    # classification use f1_score; regression use r2

    res = grid_search.fit(df_ftr[ftr_cols].values, df_ftr[target].values)
    utils.saveData(res, './model/machinelearning/cv_rf_gridsearch.pkl')

    mdl = RandomForestRegressor(n_estimators=500, max_depth=5, max_features=5, min_samples_split=10,
                                min_samples_leaf=10, max_samples=0.1)

    df_train = df_ftr.loc[df_ftr.index < pd.to_datetime('2019-5-1')]
    mdl.fit(df_train[ftr_cols], df_train[target])
    df_test = df_ftr.loc[df_ftr.index >= pd.to_datetime('2019-5-1')]
    df_pred = mdl.predict(df_test[ftr_cols])
    df_test['pred'] = df_pred

    r2_score(df_test[target], df_test['pred'])
    df_test[[target, 'pred']].corr()
    df_test.pred.hist(bins=20)
    df_test[target].hist(bins=20)

    df_test['pred_label'] = (df_test.pred > 0.03) * 1.0
    df_test['target_label'] = (df_test[target] > 0.03) * 1.0

    print(classification_report(df_test['target_label'], df_test['pred_label']))


