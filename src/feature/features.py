import pandas as pd
import talib as ta
import numpy as np

def build_talib_factors(df_ftr, tp=10):
    tporg = tp
    mul = 2
    rtn_divisor = [1, 1 / 8]
    # momentum
    df_ftr['BOP'] = ta.BOP(df_ftr.open.values, df_ftr.high.values, df_ftr.low.values, df_ftr.close.values)
    # volatility
    df_ftr['TRANGE'] = ta.TRANGE(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values)
    # volume
    df_ftr['AD'] = ta.AD(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, df_ftr.volume.values)

    #    df_ftr['AD_ANGLE']=ta.LINEARREG_ANGLE(df_ftr['AD'].values, timeperiod=tp) too little variation

    # overlap
    df_ftr['OBV'] = ta.OBV(df_ftr.close.values, df_ftr.volume.values)


    for i in range(len(rtn_divisor)):
        tp = int(tporg // rtn_divisor[i])
        if tp <= 3:
            continue
        x = str(i)
        ######## self defined
        df_ftr['rtn_disper'] = df_ftr['high'] - df_ftr['low']
        df_ftr['rtn_disper_rolling' + x] = df_ftr['rtn_disper'].rolling(int(tp)).mean()
        ######## momentum indicators
        # see descriptoin for the values df_ftr['close_slope' + x] and others see the freq it falls into the ranges
        df_ftr['close_slope' + x] = ta.LINEARREG_SLOPE(df_ftr['close'].values, timeperiod=tp)
        df_ftr['close_slope_std' + x] = df_ftr['close_slope' + x].rolling(int(tp)).std()

        # rsi
        df_ftr['rsi' + x] = ta.RSI(df_ftr.close.values, timeperiod=tp)
        df_ftr['rsi_mean' + x] = ta.SUM(df_ftr['rsi' + x].values, timeperiod=tp) / tp

        df_ftr['storsi' + x] = (df_ftr['rsi' + x] - df_ftr['rsi' + x].rolling(tp).min()) / (
                df_ftr['rsi' + x].rolling(tp).max() - df_ftr['rsi' + x].rolling(tp).min())

        # stochastic
        df_ftr['slowk' + x], df_ftr['slowd' + x] = ta.STOCH(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values,
                                                            fastk_period=round(tp * mul), slowk_period=tp,
                                                            slowk_matype=0, slowd_period=tp,
                                                            slowd_matype=0)  # slowd is slow sto, slowk is fast sto
        df_ftr['slowj'] = (3 * df_ftr['slowd' + x]) - (2 * df_ftr['slowk' + x])


        df_ftr['fastk' + x], df_ftr['fastd' + x] = ta.STOCHF(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values,
                                                             fastk_period=tp, fastd_period=tp // mul, fastd_matype=0)

        df_ftr['mom' + x] = ta.MOM(df_ftr.close.values, timeperiod=tp)
        # directional change
        df_ftr['plus_di' + x] = ta.PLUS_DI(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)
        df_ftr['plus_dm' + x] = ta.PLUS_DM(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)
        df_ftr['MINUS_DI' + x] = ta.MINUS_DI(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)
        df_ftr['MINUS_DM' + x] = ta.MINUS_DM(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)

        df_ftr['plus_minus_di' + x] = df_ftr['plus_di' + x] - df_ftr['MINUS_DI' + x]

        df_ftr['DX' + x] = ta.DX(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)
        df_ftr['ADX' + x] = ta.ADX(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)
        df_ftr['ADXR' + x] = ta.ADXR(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)


        # MACD
        df_ftr['MACD' + x], df_ftr['macdsignal' + x], df_ftr['macdhist' + x] = ta.MACD(df_ftr.close.values,
                                                                                       fastperiod=tp,
                                                                                       slowperiod=round(tp * 2),
                                                                                       signalperiod=tp // mul)
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_average_convergence_divergence_macd


        # Aroon
        # https://tradingsim.com/blog/aroon-indicator/
        df_ftr['aroondown' + x], df_ftr['aroonup' + x] = ta.AROON(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)
        df_ftr['AROONOSC' + x] = ta.AROONOSC(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)

        # Chande Momentum Oscillator
        # https://www.investopedia.com/terms/c/chandemomentumoscillator.asp
        df_ftr['CMO' + x] = ta.CMO(df_ftr.close.values, timeperiod=tp)

        # Money Flow Index
        # http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi
        df_ftr['MFI' + x] = ta.MFI(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, df_ftr.volume.values,
                                   timeperiod=tp)
        df_ftr['MFI_slope' + x] = ta.LINEARREG_SLOPE(df_ftr['MFI' + x].values, timeperiod=tp)


        # MACD with controllable MA type
        df_ftr['macdEXT' + x], df_ftr['macdEXTsignal' + x], df_ftr['macdEXThist' + x] = ta.MACDEXT(df_ftr.close.values,
                                                                                                   fastperiod=tp // mul,
                                                                                                   fastmatype=0,
                                                                                                   slowperiod=tp,
                                                                                                   slowmatype=0,
                                                                                                   signalperiod=9,
                                                                                                   signalmatype=0)
        df_ftr['macdEXT_slope' + x] = ta.LINEARREG_SLOPE(df_ftr['macdEXT' + x].values, timeperiod=tp)

        df_ftr['macdFIX' + x], df_ftr['macdFIXsignal' + x], df_ftr['macdFIXhist' + x] = ta.MACDFIX(df_ftr.close.values,
                                                                                                   signalperiod=9)


        # Stochastic Relative Strength Index
        df_ftr['fastkRSI' + x], df_ftr['fastdRSI' + x] = ta.STOCHRSI(df_ftr.close.values, timeperiod=tp,
                                                                     fastk_period=tp // 2, fastd_period=tp // (mul * 2),
                                                                     fastd_matype=0)

        # volatility
        df_ftr['ATR' + x] = ta.ATR(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)
        df_ftr['NATR' + x] = ta.NATR(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, timeperiod=tp)

        # volume

        df_ftr['ADOSC' + x] = ta.ADOSC(df_ftr.high.values, df_ftr.low.values, df_ftr.close.values, df_ftr.volume.values,
                                       fastperiod=tp, slowperiod=tp * mul)

        df_ftr['AD_SLOPE' + x] = ta.LINEARREG_SLOPE(df_ftr['AD'].values, timeperiod=tp)
        df_ftr['AD_SLOPE_std' + x] = df_ftr['AD_SLOPE' + x].rolling(int(tp * 20)).std()



        df_ftr['OBV_slope' + x] = ta.LINEARREG_SLOPE(df_ftr['OBV'].values, timeperiod=tp)


        # cycle
        df_ftr['HT_DCPERIOD' + x] = df_ftr['HT_DCPERIOD'].pct_change(periods=int(tp)).values
        df_ftr['HT_TRENDLINE' + x] = pd.DataFrame(ta.HT_TRENDLINE(df_ftr.close.values)).pct_change(
            periods=int(tp)).values

        # statistics
        df_ftr['STDDEV' + x] = ta.STDDEV(df_ftr.close.values, timeperiod=tp, nbdev=1)
        # NbDev = How may deviations you want this function to return (normally = 1).
        df_ftr['TSF' + x] = ta.TSF(df_ftr.close.values, timeperiod=tp) / df_ftr.close - 1
        df_ftr['BETA' + x] = ta.BETA(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)
        df_ftr['LINEARREG_SLOPE' + x] = ta.LINEARREG_SLOPE(df_ftr.close.values, timeperiod=tp)
        df_ftr['CORREL' + x] = ta.CORREL(df_ftr.high.values, df_ftr.low.values, timeperiod=tp)

        # candle indicators - pattern recognition - unused features

    df_ftr = df_ftr.replace([np.inf, -np.inf], np.nan)

    col = list(df_ftr.columns.values)
    df_ftr.to_pickle(r'../data/ftr_ta.pkl')
