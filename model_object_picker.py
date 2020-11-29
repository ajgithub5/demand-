# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:02:11 2020

@author: p3000445
"""

from .Model import Model
from .Model_sma import SMA
from .Model_tes_w_damping import TES_w_damping
from .Model_tes_wo_damping import TES_wo_damping
from .Model_des_w_damping import DES_w_damping
from .Model_des_wo_damping import DES_wo_damping
from .Model_ses import SES
from .Model_baseline import Baseline
from .Model_wma12 import WMA12
from .Model_wma60 import WMA60
from .Model_weekly_trend import Weekly_Trend
from .Model_arima import Arima
from .Model_sarima import Sarima

def get_model_object(model_name):
    
    if model_name == 'SMA':
        return SMA('SMA')
    elif model_name == 'WMA12':
        return WMA12('WMA12')
    elif model_name == 'WMA60':
        return WMA60('WMA60')
    elif model_name == 'SES':
        return SES('SES')
    elif model_name == 'DES_wo_damping':
        return DES_wo_damping('DES_wo_damping')
    elif model_name == 'DES_w_damping':
        return DES_w_damping('DES_w_damping')
    elif model_name == 'TES_wo_damping':
        return TES_wo_damping('TES_wo_damping')
    elif model_name == 'TES_w_damping':
        return TES_w_damping('TES_w_damping')
    elif model_name == 'Baseline':
        return Baseline('Baseline')
    elif model_name == 'Weekly_Trend':
        return Weekly_Trend('Weekly_Trend')
    elif model_name == 'Arima':
        return Arima('Arima')
    elif model_name == 'Sarima':
        return Sarima('Sarima')
    else:
        return None