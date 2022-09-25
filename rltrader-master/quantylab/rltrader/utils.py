import time
import datetime
import numpy as np


# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"
CUSTOM_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"


def get_today_str():
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str


def get_time_str():
    return datetime.datetime.fromtimestamp(
    # 수정 전 (-)    
        # int(time.time())).strftime(FORMAT_DATETIME)
    # 수정 후 (+)    
        int(time.time())).strftime(CUSTOM_FORMAT)

def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))
