import os
import pandas as pd
import numpy as np

from quantylab.rltrader import settings
#import settings  #  data_manager.py 만 돌리고싶을때 
#삼성전자 2018~2020  2020~2022
#카카오  2018~2020  
#현대자동차 2018~2020     


COLUMNS_CHART_DATA = ['date', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_TRAINING_DATA_V1 = [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V1_1 = COLUMNS_TRAINING_DATA_V1 + [
    'inst_lastinst_ratio', 'frgn_lastfrgn_ratio',
    'inst_ma5_ratio', 'frgn_ma5_ratio',
    'inst_ma10_ratio', 'frgn_ma10_ratio',
    'inst_ma20_ratio', 'frgn_ma20_ratio',
    'inst_ma60_ratio', 'frgn_ma60_ratio',
    'inst_ma120_ratio', 'frgn_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V2 = ['per', 'pbr', 'roe'] + COLUMNS_TRAINING_DATA_V1 + [
    'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 
    'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
    'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 
    'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
]

COLUMNS_TRAINING_DATA_V3 = COLUMNS_TRAINING_DATA_V2 + [
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 
    'foreign_ma60', 'foreign_ma120',
]

COLUMNS_TRAINING_DATA_V3 = list(map(
    lambda x: x if x != 'close_lastclose_ratio' else 'diffratio', COLUMNS_TRAINING_DATA_V3))

COLUMNS_TRAINING_DATA_V4 = [
    'diffratio', 'high_close_ratio', 'low_close_ratio', 'open_lastclose_ratio', 
    'volume_lastvolume_ratio', 'trans_price_exp', 'trans_price_exp_ma5', 
    'close_ma5_ratio', 'close_ma10_ratio', 'close_ma20_ratio', 'close_ma60_ratio', 'close_ma120_ratio',
    'volume_ma5_ratio', 'volume_ma10_ratio', 'volume_ma20_ratio', 'volume_ma60_ratio', 'volume_ma120_ratio',
    'close_ubb_ratio', 'close_lbb_ratio', 'macd_signal_ratio', 'rsi', 
    'buy_strength_ma5_ratio', 'sell_strength_ma5_ratio', 'prevvalid_cnt',
    'eps_krx', 'bps_krx', 'per_krx', 'pbr_krx', 'roe_krx', 'dps_krx', 'dyr_krx', 'marketcap',
    'ind', 'ind_diff', 'ind_ma5', 'ind_ma10', 'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_diff', 'inst_ma5', 'inst_ma10', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign', 'foreign_diff', 'foreign_ma5', 'foreign_ma10', 'foreign_ma20', 'foreign_ma60', 'foreign_ma120',
    'bal_rto', 'bal_rto_diff', 'bal_rto_ma5', 'bal_rto_ma10', 'bal_rto_ma20', 'bal_rto_ma60', 'bal_rto_ma120',
    'short_ratio', 'short_ratio_diff', 'short_ratio_ma5', 'short_ratio_ma10', 'short_ratio_ma20', 'short_ratio_ma60', 'short_ratio_ma120',
    'market_kospi_diffratio', 'market_kospi_ma5_ratio', 'market_kospi_ma20_ratio', 'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio',
    'market_kosdaq_diffratio', 'market_kosdaq_ma5_ratio', 'market_kosdaq_ma20_ratio', 'market_kosdaq_ma60_ratio', 'market_kosdaq_ma120_ratio',
    'market_kospi_volume_diffratio', 'market_kospi_volume_ma5_ratio', 'market_kospi_volume_ma20_ratio', 'market_kospi_volume_ma60_ratio', 'market_kospi_volume_ma120_ratio',
    'market_kosdaq_volume_diffratio', 'market_kosdaq_volume_ma5_ratio', 'market_kosdaq_volume_ma20_ratio', 'market_kosdaq_volume_ma60_ratio', 'market_kosdaq_volume_ma120_ratio',
    'fmarket_dji_diffratio', 'fmarket_dji_ma5_ratio', 'fmarket_dji_ma20_ratio', 'fmarket_dji_ma60_ratio', 'fmarket_dji_ma120_ratio',
    'fmarket_ni225_diffratio', 'fmarket_ni225_ma5_ratio', 'fmarket_ni225_ma20_ratio', 'fmarket_ni225_ma60_ratio', 'fmarket_ni225_ma120_ratio',
    'fmarket_hsi_diffratio', 'fmarket_hsi_ma5_ratio', 'fmarket_hsi_ma20_ratio', 'fmarket_hsi_ma60_ratio', 'fmarket_hsi_ma120_ratio',
    'fmarket_dji_volume_diffratio', 'fmarket_dji_volume_ma5_ratio', 'fmarket_dji_volume_ma20_ratio', 'fmarket_dji_volume_ma60_ratio', 'fmarket_dji_volume_ma120_ratio',
    'fmarket_ni225_volume_diffratio', 'fmarket_ni225_volume_ma5_ratio', 'fmarket_ni225_volume_ma20_ratio', 'fmarket_ni225_volume_ma60_ratio', 'fmarket_ni225_volume_ma120_ratio',
    'fmarket_hsi_volume_diffratio', 'fmarket_hsi_volume_ma5_ratio', 'fmarket_hsi_volume_ma20_ratio', 'fmarket_hsi_volume_ma60_ratio', 'fmarket_hsi_volume_ma120_ratio',
    'bond_k3y_diffratio', 'bond_k3y_ma5_ratio', 'bond_k3y_ma20_ratio', 'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
    'bond_k3y_volume_diffratio', 'bond_k3y_volume_ma5_ratio', 'bond_k3y_volume_ma20_ratio', 'bond_k3y_volume_ma60_ratio', 'bond_k3y_volume_ma120_ratio',
    'interestrates_base_diffratio', 'interestrates_base_ma5_ratio', 'interestrates_base_ma20_ratio', 'interestrates_base_ma60_ratio', 'interestrates_base_ma120_ratio', 'interestrates_base',
    'interestrates_us_target_diffratio', 'interestrates_us_target_ma5_ratio', 'interestrates_us_target_ma20_ratio', 'interestrates_us_target_ma60_ratio', 'interestrates_us_target_ma120_ratio', 'interestrates_us_target',
    'interestrates_us_10y_diffratio', 'interestrates_us_10y_ma5_ratio', 'interestrates_us_10y_ma20_ratio', 'interestrates_us_10y_ma60_ratio', 'interestrates_us_10y_ma120_ratio', 'interestrates_us_10y',
    'interestrates_us_30y_diffratio', 'interestrates_us_30y_ma5_ratio', 'interestrates_us_30y_ma20_ratio', 'interestrates_us_30y_ma60_ratio', 'interestrates_us_30y_ma120_ratio', 'interestrates_us_30y',
    'commodity_aluminum_diffratio', 'commodity_aluminum_ma5_ratio', 'commodity_aluminum_ma20_ratio', 'commodity_aluminum_ma60_ratio', 'commodity_aluminum_ma120_ratio',
    'commodity_cocoa_diffratio', 'commodity_cocoa_ma5_ratio', 'commodity_cocoa_ma20_ratio', 'commodity_cocoa_ma60_ratio', 'commodity_cocoa_ma120_ratio',
    'commodity_coffee_diffratio', 'commodity_coffee_ma5_ratio', 'commodity_coffee_ma20_ratio', 'commodity_coffee_ma60_ratio', 'commodity_coffee_ma120_ratio',
    'commodity_copper_diffratio', 'commodity_copper_ma5_ratio', 'commodity_copper_ma20_ratio', 'commodity_copper_ma60_ratio', 'commodity_copper_ma120_ratio',
    'commodity_corn_diffratio', 'commodity_corn_ma5_ratio', 'commodity_corn_ma20_ratio', 'commodity_corn_ma60_ratio', 'commodity_corn_ma120_ratio',
    'commodity_cotton_diffratio', 'commodity_cotton_ma5_ratio', 'commodity_cotton_ma20_ratio', 'commodity_cotton_ma60_ratio', 'commodity_cotton_ma120_ratio',
    'commodity_gold_domestic_diffratio', 'commodity_gold_domestic_ma5_ratio', 'commodity_gold_domestic_ma20_ratio', 'commodity_gold_domestic_ma60_ratio', 'commodity_gold_domestic_ma120_ratio',
    'commodity_gold_world_diffratio', 'commodity_gold_world_ma5_ratio', 'commodity_gold_world_ma20_ratio', 'commodity_gold_world_ma60_ratio', 'commodity_gold_world_ma120_ratio',
    'commodity_heating_oil_diffratio', 'commodity_heating_oil_ma5_ratio', 'commodity_heating_oil_ma20_ratio', 'commodity_heating_oil_ma60_ratio', 'commodity_heating_oil_ma120_ratio',
    'commodity_iron_diffratio', 'commodity_iron_ma5_ratio', 'commodity_iron_ma20_ratio', 'commodity_iron_ma60_ratio', 'commodity_iron_ma120_ratio',
    'commodity_lead_diffratio', 'commodity_lead_ma5_ratio', 'commodity_lead_ma20_ratio', 'commodity_lead_ma60_ratio', 'commodity_lead_ma120_ratio',
    'commodity_lumber_diffratio', 'commodity_lumber_ma5_ratio', 'commodity_lumber_ma20_ratio', 'commodity_lumber_ma60_ratio', 'commodity_lumber_ma120_ratio',
    'commodity_natural_gas_diffratio', 'commodity_natural_gas_ma5_ratio', 'commodity_natural_gas_ma20_ratio', 'commodity_natural_gas_ma60_ratio', 'commodity_natural_gas_ma120_ratio',
    'commodity_nickel_diffratio', 'commodity_nickel_ma5_ratio', 'commodity_nickel_ma20_ratio', 'commodity_nickel_ma60_ratio', 'commodity_nickel_ma120_ratio',
    'commodity_oil_diesel_diffratio', 'commodity_oil_diesel_ma5_ratio', 'commodity_oil_diesel_ma20_ratio', 'commodity_oil_diesel_ma60_ratio', 'commodity_oil_diesel_ma120_ratio',
    'commodity_oil_gasoline_diffratio', 'commodity_oil_gasoline_ma5_ratio', 'commodity_oil_gasoline_ma20_ratio', 'commodity_oil_gasoline_ma60_ratio', 'commodity_oil_gasoline_ma120_ratio',
    'commodity_oil_wti_diffratio', 'commodity_oil_wti_ma5_ratio', 'commodity_oil_wti_ma20_ratio', 'commodity_oil_wti_ma60_ratio', 'commodity_oil_wti_ma120_ratio',
    'commodity_orange_juice_diffratio', 'commodity_orange_juice_ma5_ratio', 'commodity_orange_juice_ma20_ratio', 'commodity_orange_juice_ma60_ratio', 'commodity_orange_juice_ma120_ratio',
    'commodity_palladium_diffratio', 'commodity_palladium_ma5_ratio', 'commodity_palladium_ma20_ratio', 'commodity_palladium_ma60_ratio', 'commodity_palladium_ma120_ratio',
    'commodity_platinum_diffratio', 'commodity_platinum_ma5_ratio', 'commodity_platinum_ma20_ratio', 'commodity_platinum_ma60_ratio', 'commodity_platinum_ma120_ratio',
    'commodity_rice_diffratio', 'commodity_rice_ma5_ratio', 'commodity_rice_ma20_ratio', 'commodity_rice_ma60_ratio', 'commodity_rice_ma120_ratio',
    'commodity_silver_diffratio', 'commodity_silver_ma5_ratio', 'commodity_silver_ma20_ratio', 'commodity_silver_ma60_ratio', 'commodity_silver_ma120_ratio',
    'commodity_soybean_diffratio', 'commodity_soybean_ma5_ratio', 'commodity_soybean_ma20_ratio', 'commodity_soybean_ma60_ratio', 'commodity_soybean_ma120_ratio',
    'commodity_soybean_gourd_diffratio', 'commodity_soybean_gourd_ma5_ratio', 'commodity_soybean_gourd_ma20_ratio', 'commodity_soybean_gourd_ma60_ratio', 'commodity_soybean_gourd_ma120_ratio',
    'commodity_soybean_milk_diffratio', 'commodity_soybean_milk_ma5_ratio', 'commodity_soybean_milk_ma20_ratio', 'commodity_soybean_milk_ma60_ratio', 'commodity_soybean_milk_ma120_ratio',
    'commodity_sugar11_diffratio', 'commodity_sugar11_ma5_ratio', 'commodity_sugar11_ma20_ratio', 'commodity_sugar11_ma60_ratio', 'commodity_sugar11_ma120_ratio',
    'commodity_tin_diffratio', 'commodity_tin_ma5_ratio', 'commodity_tin_ma20_ratio', 'commodity_tin_ma60_ratio', 'commodity_tin_ma120_ratio',
    'commodity_wheat_diffratio', 'commodity_wheat_ma5_ratio', 'commodity_wheat_ma20_ratio', 'commodity_wheat_ma60_ratio', 'commodity_wheat_ma120_ratio',
    'commodity_zinc_diffratio', 'commodity_zinc_ma5_ratio', 'commodity_zinc_ma20_ratio', 'commodity_zinc_ma60_ratio', 'commodity_zinc_ma120_ratio',
    'gsci_diffratio', 'gsci_ma5_ratio', 'gsci_ma20_ratio', 'gsci_ma60_ratio', 'gsci_ma120_ratio',
    'fx_usdkrw_diffratio', 'fx_usdkrw_ma5_ratio', 'fx_usdkrw_ma20_ratio', 'fx_usdkrw_ma60_ratio', 'fx_usdkrw_ma120_ratio',
    'fx_eurkrw_diffratio', 'fx_eurkrw_ma5_ratio', 'fx_eurkrw_ma20_ratio', 'fx_eurkrw_ma60_ratio', 'fx_eurkrw_ma120_ratio',
    'fx_jpykrw_diffratio', 'fx_jpykrw_ma5_ratio', 'fx_jpykrw_ma20_ratio', 'fx_jpykrw_ma60_ratio', 'fx_jpykrw_ma120_ratio',
    'fx_cnykrw_diffratio', 'fx_cnykrw_ma5_ratio', 'fx_cnykrw_ma20_ratio', 'fx_cnykrw_ma60_ratio', 'fx_cnykrw_ma120_ratio',
    'fx_hkdkrw_diffratio', 'fx_hkdkrw_ma5_ratio', 'fx_hkdkrw_ma20_ratio', 'fx_hkdkrw_ma60_ratio', 'fx_hkdkrw_ma120_ratio',
    'dx_diffratio', 'dx_ma5_ratio', 'dx_ma20_ratio', 'dx_ma60_ratio', 'dx_ma120_ratio',
    'dx_volume_diffratio', 'dx_volume_ma5_ratio', 'dx_volume_ma20_ratio', 'dx_volume_ma60_ratio', 'dx_volume_ma120_ratio',
    'bdi_diffratio', 'bdi_ma5_ratio', 'bdi_ma20_ratio', 'bdi_ma60_ratio', 'bdi_ma120_ratio',
    'sox_diffratio', 'sox_ma5_ratio', 'sox_ma20_ratio', 'sox_ma60_ratio', 'sox_ma120_ratio',
    'vix_diffratio', 'vix_ma5_ratio', 'vix_ma20_ratio', 'vix_ma60_ratio', 'vix_ma120_ratio',
    'msci_world_diffratio', 'msci_world_ma5_ratio', 'msci_world_ma20_ratio', 'msci_world_ma60_ratio', 'msci_world_ma120_ratio',
    'msci_acwi_diffratio', 'msci_acwi_ma5_ratio', 'msci_acwi_ma20_ratio', 'msci_acwi_ma60_ratio', 'msci_acwi_ma120_ratio',
    'msci_em_diffratio', 'msci_em_ma5_ratio', 'msci_em_ma20_ratio', 'msci_em_ma60_ratio', 'msci_em_ma120_ratio',
    'msci_korea_diffratio', 'msci_korea_ma5_ratio', 'msci_korea_ma20_ratio', 'msci_korea_ma60_ratio', 'msci_korea_ma120_ratio',
    'msci_usa_diffratio', 'msci_usa_ma5_ratio', 'msci_usa_ma20_ratio', 'msci_usa_ma60_ratio', 'msci_usa_ma120_ratio',
    'msci_china_diffratio', 'msci_china_ma5_ratio', 'msci_china_ma20_ratio', 'msci_china_ma60_ratio', 'msci_china_ma120_ratio',
    'msci_japan_diffratio', 'msci_japan_ma5_ratio', 'msci_japan_ma20_ratio', 'msci_japan_ma60_ratio', 'msci_japan_ma120_ratio',
    'msci_hongkong_diffratio', 'msci_hongkong_ma5_ratio', 'msci_hongkong_ma20_ratio', 'msci_hongkong_ma60_ratio', 'msci_hongkong_ma120_ratio',
    'msci_uk_diffratio', 'msci_uk_ma5_ratio', 'msci_uk_ma20_ratio', 'msci_uk_ma60_ratio', 'msci_uk_ma120_ratio',
    'msci_france_diffratio', 'msci_france_ma5_ratio', 'msci_france_ma20_ratio', 'msci_france_ma60_ratio', 'msci_france_ma120_ratio',
    'msci_germany_diffratio', 'msci_germany_ma5_ratio', 'msci_germany_ma20_ratio', 'msci_germany_ma60_ratio', 'msci_germany_ma120_ratio',
]
#remove diff_ratio, ma5, ma10 for all features from COLUMNS_TRAINING_DATA_V3 and
COLUMNS_CUSTOM = ['per', 'pbr', 'roe'] + [
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
]+[
    'market_kospi_ma20_ratio', 'market_kospi_ma60_ratio', 'market_kospi_ma120_ratio', 
     'bond_k3y_ma20_ratio', 'bond_k3y_ma60_ratio', 'bond_k3y_ma120_ratio',
]+[
    'ind',  'ind_ma20', 'ind_ma60', 'ind_ma120',
    'inst', 'inst_ma20', 'inst_ma60', 'inst_ma120',
    'foreign',  'foreign_ma20', 'foreign_ma60', 'foreign_ma120',
]

COLUMNS_CUSTOM = list(map(
    lambda x: x if x != 'close_lastclose_ratio' else 'diffratio', COLUMNS_CUSTOM))

COLUMNS_ETF = ['perb', 'bw', 'MACD_ratio', 'RSI','slow_k','slow_d' ] +[
    'open_lastclose_ratio', 'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'close_ma5_ratio', 'volume_ma5_ratio',
    'close_ma10_ratio', 'volume_ma10_ratio',
    'close_ma20_ratio', 'volume_ma20_ratio',
    'close_ma60_ratio', 'volume_ma60_ratio',
    'close_ma120_ratio', 'volume_ma120_ratio',
] 

def preprocess_etf(data):
    '''
    bband, macd, rsi, stochastic 추가.  macd는 MACD_ratio 만 사용
    (나머지는 백분율 데이터라 ratio 변환 불필요)
    '''
    # 분모 0 인경우 안만들기 위해서
    DIV0 = 0.01


    def bband(df, window = 20, k = 2):
        df['mbb'] = df['close'].rolling(window).mean()
        df['stddev'] = df['close'].rolling(window).std()
        df['ubb'] = df['mbb'] + 2*df['stddev']
        df['lbb'] = df['mbb'] - 2*df['stddev'] 
        df['perb'] = (df['close']-df['lbb']) / (df['ubb']-df['lbb'] + DIV0)
        df['bw'] = (df['ubb']-df['lbb']) / (df['mbb'] + DIV0)

        return df

    def macd(df):
        macd_short, macd_long, macd_signal = 12,26,9
        df['MACD_short'] = df['close'].rolling(macd_short).mean()
        df['MACD_long'] = df['close'].rolling(macd_long).mean()
        df['MACD'] = df.apply(lambda x : (x['MACD_short']-x['MACD_long']), axis=1)
        df['MACD_signal'] = df['MACD'].rolling(macd_signal).mean()
        df['MACD_ratio'] = (df['MACD']/(df['MACD_signal']+ DIV0))

        return df

    def rsi(df, window = 14):
        delta =df['close'].diff(1)  #df['c_diff'] = df['close']- df['close'].shift(1)
        delta = delta.dropna() #or delta[1:]

        rsi_U = delta.copy()
        rsi_D = delta.copy()
        rsi_U[rsi_U<0] =0   #U = np.where(df.diff(1)['close'] >0, df.diff(1)['close'],0)
        rsi_D[rsi_D>0] =0   #D = np.where(df.diff(1)['close'] <0, df.diff(1)['close']*(-1),0)
        df['U'] = rsi_U
        df['D'] = rsi_D

        
        AU = df['U'].rolling(window).mean()
        AD = df['D'].rolling(window).mean()
        RSI = (AU / (AU+AD))   # *100 하면 백분율
        df['RSI'] = RSI

        return df
    
    def stochas(df, sto_n = 14, sto_m=1, sto_t=3):
        #슬로우 스토캐스틱만 사용
        ndays_high = df['high'].rolling(window =sto_n, min_periods=1).max()
        ndays_low = df['low'].rolling(window=sto_n,min_periods=1).min()

        ndays_high.fillna(0)
        ndays_low.fillna(0)

        df['Stochastic'] = ((df['close']-ndays_low)/(ndays_high-ndays_low+ DIV0))*100
        df['slow_k'] = df['Stochastic'].rolling(sto_m).mean()
        df['slow_d'] = df['slow_k'].rolling(sto_t).mean()

        return df


    windows = [5, 10, 20, 60, 120]
    #close_ma5,10,20,60,120 volumn_ma5,10,20,60,120, open_lastclose, high_close, 
    #low_close, close_lastclose, volumn_lastvolume 추가   Close  Open High Low
    for window in windows: 
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['Volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}'] 
        data[f'volume_ma{window}_ratio'] = \
            (data['Volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values 
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['Volume'][1:].values - data['Volume'][:-1].values) 
        / data['Volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values)
    
    data = bband(data)
    data = macd(data)
    data = rsi(data)
    data = stochas(data)
    
    for keyword in ['perb', 'bw', 'MACD_ratio', 'RSI','slow_k','slow_d' ]:
        for window in windows :
            data[f'{keyword}_ma{window}'] = data[keyword].rolling(window).mean()
    
    new_column_list = [] + COLUMNS_ETF
    for keyword in ['perb', 'bw', 'MACD_ratio', 'RSI','slow_k','slow_d' ]:
        for window in windows :    
            new_column_list.append(f'{keyword}_ma{window}')

    return data, new_column_list

def preprocess(data, ver='v1'):
    windows = [5, 10, 20, 60, 120]
    for window in windows:
        data[f'close_ma{window}'] = data['close'].rolling(window).mean()
        data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
        data[f'close_ma{window}_ratio'] = \
            (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
        data[f'volume_ma{window}_ratio'] = \
            (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
    data['open_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'open_lastclose_ratio'] = \
        (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
    data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
    data['close_lastclose_ratio'] = np.zeros(len(data))
    data.loc[1:, 'close_lastclose_ratio'] = \
        (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
    data['volume_lastvolume_ratio'] = np.zeros(len(data))
    data.loc[1:, 'volume_lastvolume_ratio'] = (
        (data['volume'][1:].values - data['volume'][:-1].values) 
        / data['volume'][:-1].replace(to_replace=0, method='ffill')\
            .replace(to_replace=0, method='bfill').values
    )
    

    # 기관순매수 inst
    # 외국인 순매수 frgn
    if ver == 'v1.1':
        for window in windows:
            data[f'inst_ma{window}'] = data['close'].rolling(window).mean()
            data[f'frgn_ma{window}'] = data['volume'].rolling(window).mean()
            data[f'inst_ma{window}_ratio'] = \
                (data['close'] - data[f'inst_ma{window}']) / data[f'inst_ma{window}']
            data[f'frgn_ma{window}_ratio'] = \
                (data['volume'] - data[f'frgn_ma{window}']) / data[f'frgn_ma{window}']
        data['inst_lastinst_ratio'] = np.zeros(len(data))
        data.loc[1:, 'inst_lastinst_ratio'] = (
            (data['inst'][1:].values - data['inst'][:-1].values)
            / data['inst'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )
        data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
        data.loc[1:, 'frgn_lastfrgn_ratio'] = (
            (data['frgn'][1:].values - data['frgn'][:-1].values)
            / data['frgn'][:-1].replace(to_replace=0, method='ffill')\
                .replace(to_replace=0, method='bfill').values
        )

    return data




# def preprocess(data, ver='v1'):
#     windows = [5, 10, 20, 60, 120]
#     for window in windows:
#         data[f'close_ma{window}'] = data['close'].rolling(window).mean()
#         data[f'volume_ma{window}'] = data['volume'].rolling(window).mean()
#         data[f'close_ma{window}_ratio'] = \
#             (data['close'] - data[f'close_ma{window}']) / data[f'close_ma{window}']
#         data[f'volume_ma{window}_ratio'] = \
#             (data['volume'] - data[f'volume_ma{window}']) / data[f'volume_ma{window}']
        
#     data['open_lastclose_ratio'] = np.zeros(len(data))
#     data.loc[1:, 'open_lastclose_ratio'] = \
#         (data['open'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
#     data['high_close_ratio'] = (data['high'].values - data['close'].values) / data['close'].values
#     data['low_close_ratio'] = (data['low'].values - data['close'].values) / data['close'].values
#     data['close_lastclose_ratio'] = np.zeros(len(data))
#     data.loc[1:, 'close_lastclose_ratio'] = \
#         (data['close'][1:].values - data['close'][:-1].values) / data['close'][:-1].values
#     data['volume_lastvolume_ratio'] = np.zeros(len(data))
#     data.loc[1:, 'volume_lastvolume_ratio'] = (
#         (data['volume'][1:].values - data['volume'][:-1].values) 
#         / data['volume'][:-1].replace(to_replace=0, method='ffill')\
#             .replace(to_replace=0, method='bfill').values
#     )
    

#     # 기관순매수 inst
#     # 외국인 순매수 frgn
#     if ver == 'v1.1':
#         for window in windows:
#             data[f'inst_ma{window}'] = data['close'].rolling(window).mean()
#             data[f'frgn_ma{window}'] = data['volume'].rolling(window).mean()
#             data[f'inst_ma{window}_ratio'] = \
#                 (data['close'] - data[f'inst_ma{window}']) / data[f'inst_ma{window}']
#             data[f'frgn_ma{window}_ratio'] = \
#                 (data['volume'] - data[f'frgn_ma{window}']) / data[f'frgn_ma{window}']
#         data['inst_lastinst_ratio'] = np.zeros(len(data))
#         data.loc[1:, 'inst_lastinst_ratio'] = (
#             (data['inst'][1:].values - data['inst'][:-1].values)
#             / data['inst'][:-1].replace(to_replace=0, method='ffill')\
#                 .replace(to_replace=0, method='bfill').values
#         )
#         data['frgn_lastfrgn_ratio'] = np.zeros(len(data))
#         data.loc[1:, 'frgn_lastfrgn_ratio'] = (
#             (data['frgn'][1:].values - data['frgn'][:-1].values)
#             / data['frgn'][:-1].replace(to_replace=0, method='ffill')\
#                 .replace(to_replace=0, method='bfill').values
#         )

#     return data


def load_data(code, date_from, date_to, ver='v2'):

    print('\n','-'*50 , "\tDEBUGGING..  in [data_manager.py]\t", '-'*50 )
    print(f"\tDEBUG // Loading {ver} datasets in load_data()")
    if ver in ['v3', 'v4']:
        return load_data_v3_v4(code, date_from, date_to, ver)
    if ver in ['custom']:
        return load_data_custom(code, date_from, date_to, ver)
    if ver in ['etf']:
        return load_data_etf(code, date_from, date_to, ver)

    header = None if ver == 'v1' else 0
    df = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, f'{code}.csv'),
        thousands=',', header=header, converters={'date': lambda x: str(x)})
    if ver == 'v1':
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    # 데이터 전처리
    df = preprocess(df)
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    df = df.fillna(method='ffill').reset_index(drop=True)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = None
    if ver == 'v1':
        training_data = df[COLUMNS_TRAINING_DATA_V1]
    elif ver == 'v1.1':
        training_data = df[COLUMNS_TRAINING_DATA_V1_1]
    elif ver == 'v2':
        df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
        training_data = df[COLUMNS_TRAINING_DATA_V2]
        training_data = training_data.apply(np.tanh)
    else:
        raise Exception('Invalid version.')
    
    return chart_data, training_data


def load_data_v3_v4(code, date_from, date_to, ver):

    columns = None
    if ver == 'v3':
        columns = COLUMNS_TRAINING_DATA_V3
    elif ver == 'v4':
        columns = COLUMNS_TRAINING_DATA_V4
        
    print(f"\tDEBUG // Loading {ver} datasets in load_data_v3_v4()")
    print(f"\tDEBUG // COLUMNS Feature # : {len(columns)}")

    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})

    '''  marketfeatures.csv
date	market_kospi_ma5_ratio	market_kospi_ma20_ratio	market_kospi_ma60_ratio	market_kospi_ma120_ratio	bond_k3y_ma5_ratio	bond_k3y_ma20_ratio	bond_k3y_ma60_ratio	bond_k3y_ma120_ratio
20180101	0.011111398	-0.000270659	-0.009637629	0.011877629	0.0004253	-0.001435025	-0.002387123	-0.006521886
20180102	0.011962396	0.004589309	-0.005469457	0.016519852	0.00018492	-0.001440031	-0.002541096	-0.00670917
...
20211231	-0.00765908	-0.005824218	-0.001232365	-0.032829967	-0.000273598	0.000276029	0.002615838	-0.002569326
    '''

    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break
    
    ''' 005930 삼성전자.csv
date	open	high	low	close	volume	per	pbr	roe	open_lastclose_ratio	high_close_ratio	low_close_ratio	diffratio	volume_lastvolume_ratio	close_ma5_ratio	volume_ma5_ratio	close_ma10_ratio	volume_ma10_ratio	close_ma20_ratio	volume_ma20_ratio	close_ma60_ratio	volume_ma60_ratio	close_ma120_ratio	volume_ma120_ratio	ind	ind_diff	ind_ma5	ind_ma10	ind_ma20	ind_ma60	ind_ma120	inst	inst_diff	inst_ma5	inst_ma10	inst_ma20	inst_ma60	inst_ma120	foreign	foreign_diff	foreign_ma5	foreign_ma10	foreign_ma20	foreign_ma60	foreign_ma120
20180102	51380	51400	50780	51020	8474250	10.37	1.46	14.81	0.008241758	0.00744806	-0.004704038	0.001177394	-0.056891975	0.023511475	-0.235766412	0.015040586	-0.265695045	0.00522116	-0.276464222	-0.04320158	-0.229900557	-0.000607236	-0.24379874	0.153925126	0.254270685	-0.063636467	-0.056330855	-0.019335593	0.015339645	0.005371977	-0.138094817	-0.163224332	0.037910983	0.108695817	0.085087607	-0.023288027	-0.001560204	-0.106316193	-0.148278477	-0.016003263	-0.096875576	-0.111401795	-0.044176488	-0.056207371
20180103	52540	52560	51420	51620	10013500	10.37	1.46	14.81	0.029792238	0.018209996	-0.003874467	0.011760094	0.181638493	0.027631789	-0.07720989	0.024938448	-0.093718889	0.016261763	-0.134493047	-0.032058454	-0.087239423	0.010479335	-0.106026831	-0.022554551	-0.176479677	-0.004994511	-0.064089524	-0.025702235	0.013545216	0.004866674	-0.237494383	-0.099399566	-0.045259544	0.07688029	0.069618092	-0.025208505	-0.003509053	0.228486543	0.334802736	0.005437931	-0.055945461	-0.089089143	-0.039823285	-0.053365995
...
20211230	78900	79500	78100	78300	14236700	10.37	1.46	14.81	0.001269036	0.01532567	-0.002554278	-0.006345178	-0.280785681	-0.016578749	-0.052498279	-0.009612952	-0.014277662	0.002817623	-0.045470082	0.065402758	-0.049241494	0.039173616	-0.126407884	0.166844845	-0.122886269	-0.10336186	-0.128263928	-0.08818603	-0.025604181	0.023709155	-0.210268672	0.096868482	0.062347527	0.046942104	0.011723158	0.010516353	-0.002966207	0.040313767	0.025278248	0.027736864	0.077358959	0.074475248	0.014208339	-0.022601413
    '''

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)

    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    print(f"DEBUG // df_market_features.shape = {df_marketfeatures.shape}, df_stockfeatures.shape = {df_stockfeatures.shape}")
    print(f"DEBUG // df_merge.shape = {df.shape}")
    

    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
# 수정 전(-)
    #df = df.fillna(method='ffill').reset_index(drop=True)          
# 수정 후(+)
    # print(f"DEBUG // df_merge(data filtered).shape = {df.shape}")
    # print("-"*50,\
    #             "\n결측치(NaN, None) 개수 : ",df.isnull().sum().sum(),"정상값 개수 : ", df.notnull().sum().sum(), \
    #             "\n","-"*50,
    #             "\n결측치 리스트 : \n",df.isnull().sum(), "\n","-"*50 )
    # print("\nfilling Nan value with [ffill] in Dataframe")
    # print("\nfilling Nan value with [interpolate] in Dataframe")
    df_fillna = df.fillna(method='ffill').reset_index(drop=True) 
    df_interpolate = df.interpolate() 
    
    df = df_interpolate

    '''
    df.fillna(ffill : 결측값을 앞의 값으로 채우기) 채우는 방법 종류 : https://rfriend.tistory.com/262
    interpolate 방식으로 바꾸는게 안전할 것 같기도 https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html
    Python pandas에서는 결측값을 'NaN' 으로 표기하며, 'None'도 결측값으로 인식
    '''

    # 데이터 조정
    df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)

    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]

    # 학습 데이터 분리
    training_data = df[columns]

    return chart_data, training_data

#python main.py --mode train --ver v3 --name test1 --stock_code 005930 --rl_method a2c --net dnn 
#--start_date 20180101 --end_date 20191231

def load_data_custom(code, date_from, date_to, ver):

    columns = COLUMNS_CUSTOM
    print(f"\tDEBUG // Loading {ver} datasets in load_data_custom()")
    print(f"\tDEBUG // COLUMNS Feature # : {len(columns)}")

    ver = 'v3'
    # 시장 데이터
    df_marketfeatures = pd.read_csv(
        os.path.join(settings.BASE_DIR, 'data', ver, 'marketfeatures.csv'), 
        thousands=',', header=0, converters={'date': lambda x: str(x)})
    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'date': lambda x: str(x)})
            break

    # 시장 데이터와 종목 데이터 합치기
    df = pd.merge(df_stockfeatures, df_marketfeatures, on='date', how='left', suffixes=('', '_dup'))
    df = df.drop(df.filter(regex='_dup$').columns.tolist(), axis=1)
    # 날짜 오름차순 정렬
    df = df.sort_values(by='date').reset_index(drop=True)

    print(f"\tDEBUG // df_market_features.shape : {df_marketfeatures.shape} + df_stockfeatures.shape : {df_stockfeatures.shape}\
                     \n\t\t= df_merge.shape : {df.shape}")
    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]
    print(f"\tDEBUG // df_merge(data filtered from start_date to end_date).shape = {df.shape}")
    print("\t\tdf_merge 의 결측치(NaN, None) 총 개수 : ",df.isnull().sum().sum(), "(filling 안함)")  
    #df_fillna = df.fillna(method='ffill').reset_index(drop=True)   #ffill 방식
    df_interpolate = df.interpolate() # interpolate 방식
    print("\t\tdf_merge 의 결측치(NaN, None) 총 개수 : ",df_interpolate.isnull().sum().sum(), "(filling NaN with [interpolate]) ")
    
    df = df_interpolate
    # 데이터 조정
    df.loc[:, ['per', 'pbr', 'roe']] = df[['per', 'pbr', 'roe']].apply(lambda x: x / 100)
    # 차트 데이터 분리
    chart_data = df[COLUMNS_CHART_DATA]
    # 학습 데이터 분리
    training_data = df[columns]

    print(f"\tDEBUG // Final dataset :  \n\t\tchart_data = df[COLUMNS_CHART_DATA], training_data = df[columns]")
    print(f"\t\tchart_data.shape {chart_data.shape}, training_data.shape= {training_data.shape}")
    print(f"\n\tSample of chart_data")
    print(chart_data.head(5))
    print(f"\n\tSample of training_data")
    print(training_data.head(5))

    return chart_data, training_data

def load_data_etf(code, date_from, date_to, ver):

    columns_chart_data = COLUMNS_CHART_DATA #['Date', 'Open', 'High', 'Low', 'Close', 'Volume']랑 다름(소문자)
    columns_training_data = COLUMNS_ETF
    print(f"\tDEBUG // Loading {ver} datasets in load_data_etf()")
    print(f"\tDEBUG // train_data COLUMNS # : {len(columns_training_data)}")

    ver = 'etf'

    # 종목 데이터
    df_stockfeatures = None
    for filename in os.listdir(os.path.join(settings.BASE_DIR, 'data', ver)):
        if filename.startswith(code):
            df_stockfeatures = pd.read_csv(
                os.path.join(settings.BASE_DIR, 'data', ver, filename), 
                thousands=',', header=0, converters={'Date': lambda x: str(x)})
            break

    df = df_stockfeatures
    df['date']= df['Date']
    df['open'] = df['Open']
    df['close']= df['Close']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['volume']= df['Volume']

    # 아직 시장 데이터(환율) 준비 안됬으므로 pd.merge 안함 

    df, TRAINING_COLUMN = preprocess_etf(df)
    df = df.sort_values(by='date').reset_index(drop=True)       # 날짜 오름차순 정렬

    
    # 기간 필터링
    df['date'] = df['date'].str.replace('-', '')
    df = df[(df['date'] >= date_from) & (df['date'] <= date_to)]

    print("\t\tETF 데이터의 결측치(NaN, None) 총 개수 : ",df.isnull().sum().sum(), "(filling 안함)")  
    df_interpolate = df.fillna(method='ffill').reset_index(drop=True)  
    df_interpolate = df.fillna(method='bfill').reset_index(drop=True) # interpolate는 앞에 값이 없으면 안채워짐
    print("\t\t결측치 제거 후 총 개수 : ",df_interpolate.isnull().sum().sum(), "(filling NaN with [ffill, bfill]) ")
    
    df = df_interpolate
    
    df.to_csv(os.path.join(settings.BASE_DIR, 'data', 'etf', f'preprocessed_{code}.csv'))


    # 차트 데이터 분리
    chart_data = df[columns_chart_data]
    # 학습 데이터 분리
    training_data = df[TRAINING_COLUMN]

    print(f"\tDEBUG // Final dataset :  ")
    print(f"\n\t\tchart_data.shape {chart_data.shape}, training_data.shape= {training_data.shape}")
    print(f"\n\tSample of chart_data")
    print(chart_data.head(15))
    print(f"\n\tSample of training_data")
    print(training_data.head(15))

    print(f"\n\tchart_data columns :")
    print(chart_data.columns)
    print(f"\n\ttraining_data columns :")
    print(training_data.columns)
    print(f"\n\tDEBUG // END of data_manager.py ")

    return chart_data, training_data





if __name__ == '__main__':
    print("DEBUG //  here is data_manager.py.__main__  ")
    chart_data, training_data = load_data_custom( "005930", "20180101", "20191231", 'custom')

    print("-"*50,"\nDEBUG //  here is END of data_manager.py.__main__  ")


