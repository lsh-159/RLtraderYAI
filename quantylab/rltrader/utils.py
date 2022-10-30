import time
import datetime
import numpy as np

from quantylab.rltrader import settings


# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"
CUSTOM_FORMAT = "%Y-%m-%d_%H-%M-%S"
CUSTOM_FORMAT2 = "%y%m%d_%H-%M"


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
        int(time.time())).strftime(CUSTOM_FORMAT2)

def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))

def exploration_function(start_eps, total_epoches, curr_epoch):
    '''
    if epoch <1000 :
        return epsilon = (total_epoches - current_epoch) / total_epoches (linear하게 감소)
    else if epoch >1000 :
        return settings.eps (고정된 탐험률 = 0.1)
    '''
    epsilon = start_eps * (1 - (curr_epoch / (total_epoches - 1)))
    if curr_epoch > 1000 and epsilon < settings.AFTER_EPOCH1000_eps :
        epsilon = settings.AFTER_EPOCH1000_eps
    return epsilon

def reward_shaping(reward_, prev_pt, curr_pt, version=0):
    # r = ln(pt/(p_t-1))
    if version:
        reward = np.log(0.001 + curr_pt/(prev_pt+0.001))
    else :
        reward = reward_
    


    return reward