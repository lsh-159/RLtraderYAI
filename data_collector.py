import os
import sys
import logging
import argparse
import json

from quantylab.rltrader import settings
from quantylab.rltrader import utils

import pandas as pd
import matplotlib.pyplot as plt
import FinanceDataReader as fdr
#import talib as tl  #설치5분, 그래도 코랩에서 에러
#import FinanceDataReader as fdr
#import yfinance as yf
#from pykrx import stock
#from pykrx import bond


'''다음이 깔려야 실행됨
!pip install finance-datareader 
!pip install bs4
'''

ETF_LIST = ['KS11','KQ11','KS50', 'KS100', 'KRX100', 'KS200','DJI','IXIC', 'US500','RUTNU',\
                    'VIX', 'JP225','STOXX50', 'HK50', 'CSI300', 'TWII', 'HNX30', 'SSEC', 'UK100', 'DE30', 'FCHI'] 

'''
한국 주요 지수            미국 주요 지수                              국가별 대표 지수

심볼	설명                심볼	설명                                  심볼	설명
KS11	KOSPI 지수          DJI	다우존스 지수                           JP225	닛케이 225 선물
KQ11	KOSDAQ 지수         IXIC	나스닥 종합 지수                      STOXX50	유럽 STOXX 50
KS50	KOSPI 50 지수       US500	S&P 500 지수                          HK50	항셍 지수
KS100	KOSPI 100           RUTNU	러셀 2000 (US Small Cap 2000)         CSI300	CSI 300 (중국)
KRX100	KRX 100           VIX	CBOE 변동성 지수 (공포지수)             TWII	대만 가권 지수
KS200	코스피 200                                                      HNX30	베트남 하노이
                                                                      SSEC	상해 종합
                                                                      UK100	영국 FTSE
                                                                      DE30	독일 DAX 30
                                                                      FCHI	프랑스 CAC 40
'''


def making_csv(code) :
    if code not in ETF_LIST:
        print('ERROR : --code is not in ETF_LIST')
        exit(0)

    print(code)
    df = fdr.DataReader(code)
    df_valid = df[df['Volume']>0]
    df_valid.to_csv(os.path.join(settings.BASE_DIR, 'data', 'etf') + f'{code}.csv')

    print('------------original etf data------------')
    print(df)
    print('------------delete volume ==0 -----------')
    print(df_valid)

def making_csv_all():
    for code in ETF_LIST:
        print(code)
        df = fdr.DataReader(code)
        df_valid = df[df['Volume']>0]
        df_valid.to_csv(os.path.join(settings.BASE_DIR, 'data', 'etf') + f'{code}.csv')
        print('-'*50)
        print(df_valid)
    
if __name__ == '__main__':
    print("DEBUG //  here is data_collector.py.__main__  ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--code',  default='all')
    args = parser.parse_args()

    save_path = os.path.join(settings.BASE_DIR, 'data','etf')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
		
    if args.code=='all':
        making_csv_all()
    else :
        making_csv(args.code)

    print("-"*50,"\nDEBUG //  here is END of data_collector.py.__main__  ")

#python data_collector.py --code all 

