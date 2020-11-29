# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 06:46:50 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from models.Model import Model

from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm


class Arima(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name,df):
    
        """
        An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :start_p                     : int, optional (default=2),The starting value of p, the order (or number of time lags) of the auto-regressive (“AR”) model. Must be a positive integer.
        :start_q                     : int, optional (default=2) The starting value of q, the order of the moving-average (“MA”) model. Must be a positive integer.
        :d                           : int, optional (default=None) The order of first-differencing.None (by default)
        :max_p                       : int, optional (default=5) The maximum value of p, inclusive. Must be a positive integer greater than or equal to start_p.
        :max_d                       : int, optional (default=2) The maximum value of d, or the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
        :max_q                       : int, optional (default=5) The maximum value of q, inclusive. Must be a positive integer greater than start_q.
        :seasonal                    : bool, optional (default=True) Whether to fit a seasonal ARIMA. Default is True. Note that if seasonal is True and m == 1, seasonal will be set to False.
        :stepwise                    : bool, optional (default=True) Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters. The stepwise algorithm can be significantly faster than fitting all (or a random subset of) hyper-parameter combinations and is less likely to over-fit the model.
        :suppress_warnings           : bool, optional (default=False) Many warnings might be thrown inside of statsmodels. If suppress_warnings is True, all of the warnings coming from ARIMA will be squelched.
        :error_action                : str, optional (default=’warn’) If unable to fit an ARIMA due to stationarity issues, whether to warn (‘warn’), raise the ValueError (‘raise’) or ignore (‘ignore’). Note that the default behavior is to warn, and fits that fail will be returned as None.
        :trace                       : bool or int, optional (default=False) Whether to print status on the fits. A value of False will print no debugging information. A value of True will print some. Integer values exceeding 1 will print increasing amounts of debug information at each fit.
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        #df['train'] = df['train'].replace(0,0.001)
        
        model = pm.auto_arima(df["train"][:-m],start_p=1, start_q=1,test='adf',max_p=3, max_q=3,d=None,seasonal=False, trace = True, error_action = 'ignore', suppress_warnings = True, stepwise = True)
        fit1 = model.fit(df["train"][:-m])
        params = fit1.params
        df[column_name] = np.nan
        y_fore = fit1.predict(m)
        
        #y_fore = model.forecast(m)
        
        df[column_name][:-m] = df['train'].iloc[:-m]
        #df[column_name][:-1] = y_fit
        df[column_name][-m:] = y_fore

        
        
        return df
    
    def ADE_base_model(self, column_name,df):
    
        """
        An ARIMA, or autoregressive integrated moving average, is a generalization of an autoregressive moving average (ARMA) and is fitted to time-series data in an effort to forecast future points.
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :start_p                     : int, optional (default=2),The starting value of p, the order (or number of time lags) of the auto-regressive (“AR”) model. Must be a positive integer.
        :start_q                     : int, optional (default=2) The starting value of q, the order of the moving-average (“MA”) model. Must be a positive integer.
        :d                           : int, optional (default=None) The order of first-differencing.None (by default)
        :max_p                       : int, optional (default=5) The maximum value of p, inclusive. Must be a positive integer greater than or equal to start_p.
        :max_d                       : int, optional (default=2) The maximum value of d, or the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
        :max_q                       : int, optional (default=5) The maximum value of q, inclusive. Must be a positive integer greater than start_q.
        :seasonal                    : bool, optional (default=True) Whether to fit a seasonal ARIMA. Default is True. Note that if seasonal is True and m == 1, seasonal will be set to False.
        :stepwise                    : bool, optional (default=True) Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters. The stepwise algorithm can be significantly faster than fitting all (or a random subset of) hyper-parameter combinations and is less likely to over-fit the model.
        :suppress_warnings           : bool, optional (default=False) Many warnings might be thrown inside of statsmodels. If suppress_warnings is True, all of the warnings coming from ARIMA will be squelched.
        :error_action                : str, optional (default=’warn’) If unable to fit an ARIMA due to stationarity issues, whether to warn (‘warn’), raise the ValueError (‘raise’) or ignore (‘ignore’). Note that the default behavior is to warn, and fits that fail will be returned as None.
        :trace                       : bool or int, optional (default=False) Whether to print status on the fits. A value of False will print no debugging information. A value of True will print some. Integer values exceeding 1 will print increasing amounts of debug information at each fit.
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        #df['train'] = df['train'].replace(0,0.001)
        
        model = pm.auto_arima(df["train"][:-m],start_p=1, start_q=1,test='adf',max_p=3, max_q=3,d=None,seasonal=False, trace = True, error_action = 'ignore', suppress_warnings = True, stepwise = True)
        fit1 = model.fit(df["train"][:-m])
        params = fit1.params
        
        df[column_name] = np.nan
        y_fit = fit1.predict_in_sample()
        y_fore = fit1.predict(m)
        
        #y_fore = model.forecast(m)
        
        df[column_name][:-m] = y_fit
        #df[column_name][:-1] = y_fit
        df[column_name][-m:] = y_fore

        
        
        return df

    def cv_search(self, df_splits,column_name, logic, df_daily, team_location, hyperparam_combinations={}):
        
        """
        performs grid-search on walk-forward splits to find the optimal hyperparameters
        
        :input df_splits             : list of walk-forward validation split dataframes
        :input column_name           : str, name of column to hold the predicted values
        :input logic                 : str, the type of approach to use for holiday inflation/deflation
        :input df_daily              : dataframe, pre-processed daily-level data
        :input team_location         : str, location of the team selected (England/Scotland/Offshore)
        :input hyperparams           : dictionary, default={}, details of hyperparameters and their corresponding ranges to consider for grid-search
            
        :returns optimal_hyperparams : dictionary, conatining mapping of hyperparmeter, and its optimal value 
        """
    
        hyperparam_combinations = None
        #print(f'hyperparams: {hyperparams}')
        optimal_hyperparams = super().cv_search(df_splits, column_name, logic, df_daily, team_location, hyperparam_combinations) 
        
        return optimal_hyperparams

    def evaluate(self, df, column_name, logic, team_name, df_daily, team_location):
        
        """
        evluates model metrics on all walk-forward split dataframes using the predictions usng the most optimal hyperparameters chosen from grid-search method
    
        :input df                    : dataframe, weekly level data
        :input column_name           : str, name of column to hold the predicted values
        :input logic                 : str, the type of approach to use for holiday inflation/deflation
        :input team_name             : str, name of the team to filter on
        :input df_daily              : dataframe pre-processed daily-level data
        :input team_location         : str, location of the team selected (England/Scotland/Offshore)
        
        :returns framework           : dataframe, containing accuracy metrics for each data point (weekly)
        :returns model_metrics       : dictionary, conatining average accuracy metrics for the model performance over the entire range of walk-forward dataframe splits
        :returns optimal_hyperparams : dictionary, conatining mapping of hyperparmeter, and its optimal value 
        
        """
        
        '''
        framework, metrics, optimal_hyperparams = super().evaluate(df, column_name, logic, team_name, df_daily, team_location)
        return framework, metrics, optimal_hyperparams
        '''
        
        result = super().evaluate(df, column_name, logic, team_name, df_daily, team_location)
        
        if result['status']:
            return result
        else:
            return {
                    'status':False,
                    'error':result['error']
                    }