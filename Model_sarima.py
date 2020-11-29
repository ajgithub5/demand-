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


class Sarima(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name,df, start_p,d,start_q,max_p,max_d,max_q,start_P,D,start_Q,max_P,max_D,max_Q,m,seasonal,test,seasonal_test,trace,error_action, suppress_warnings, stepwise):
    
        """
        SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. 
        It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input start_p                     : int, optional (default=2),The starting value of p, the order (or number of time lags) of the auto-regressive (“AR”) model. Must be a positive integer.
        :input start_q                     : int, optional (default=2) The starting value of q, the order of the moving-average (“MA”) model. Must be a positive integer.
        :input d                           : int, optional (default=None) The order of first-differencing.None (by default)
        :input max_p                       : int, optional (default=5) The maximum value of p, inclusive. Must be a positive integer greater than or equal to start_p.
        :input max_d                       : int, optional (default=2) The maximum value of d, or the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
        :input max_q                       : int, optional (default=5) The maximum value of q, inclusive. Must be a positive integer greater than start_q.
        :input start_P                     : int, optional (default=1) The starting value of P, the order of the auto-regressive portion of the seasonal model.
        :input D                           : int, optional (default=None) The order of the seasonal differencing. If None (by default, the value will automatically be selected based on the results of the seasonal_test. Must be a positive integer or None.
        :input start_Q                     : int, optional (default=1) The starting value of Q, the order of the moving-average portion of the seasonal model.
        :input max_P                       : int, optional (default=2) The maximum value of P, inclusive. Must be a positive integer greater than start_P.
        :input max_D                       : int, optional (default=1) The maximum value of D. Must be a positive integer greater than D.
        :input max_Q                       : int, optional (default=2) The maximum value of Q, inclusive. Must be a positive integer greater than start_Q.
        :input m                           : int, optional (default=1) The period for seasonal differencing, m refers to the number of periods in each season. For example, m is 4 for quarterly data, 12 for monthly data, or 1 for annual (non-seasonal) data. Default is 1. Note that if m == 1 (i.e., is non-seasonal), seasonal will be set to False.
        :input seasonal                    : bool, optional (default=True) Whether to fit a seasonal ARIMA. Default is True. Note that if seasonal is True and m == 1, seasonal will be set to False.
        :input test                        : str, optional (default=’kpss’) Type of unit root test to use in order to detect stationarity if stationary is False and d is None.
        :input seasonal_test               : str, optional (default=’ocsb’) This determines which seasonal unit root test is used if seasonal is True and D is None.
        :input stepwise                    : bool, optional (default=True) Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters. The stepwise algorithm can be significantly faster than fitting all (or a random subset of) hyper-parameter combinations and is less likely to over-fit the model.
        :input suppress_warnings           : bool, optional (default=False) Many warnings might be thrown inside of statsmodels. If suppress_warnings is True, all of the warnings coming from ARIMA will be squelched.
        :input error_action                : str, optional (default=’warn’) If unable to fit an ARIMA due to stationarity issues, whether to warn (‘warn’), raise the ValueError (‘raise’) or ignore (‘ignore’). Note that the default behavior is to warn, and fits that fail will be returned as None.
        :input trace                       : bool or int, optional (default=False) Whether to print status on the fits. A value of False will print no debugging information. A value of True will print some. Integer values exceeding 1 will print increasing amounts of debug information at each fit.
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        df['train'] = df['train'].replace(0,0.001)
        
        model = pm.auto_arima(df["train"][:-m],start_p=start_p,d=d,start_q=start_q,max_p=max_p,max_d=max_d,max_q=max_q,start_P=start_P,D=D,start_Q=start_Q,max_P=max_P,max_D=max_D,max_Q=max_Q,m=m,seasonal=seasonal,test=test,seasonal_test=seasonal_test,trace=trace,error_action=error_action, suppress_warnings=suppress_warnings, stepwise=stepwise)
        fit1 = model.fit(df["train"][:-m])
        params = fit1.params
        df[column_name] = np.nan
        #y_fit = fit1.fittedvalues
        y_fore = fit1.predict(m)
        
        #y_fore = model.forecast(m)
        
        df[column_name][:-m] = df['train'].iloc[:-m]
        #df[column_name][:-1] = y_fit
        df[column_name][-m:] = y_fore

        #print (params)
        
        return df
    
    def ADE_base_model(self, column_name,df, start_p,d,start_q,max_p,max_d,max_q,start_P,D,start_Q,max_P,max_D,max_Q,m,seasonal,test,seasonal_test,trace,error_action, suppress_warnings, stepwise):
    
        """
        SARIMA or Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component. 
        It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input start_p                     : int, optional (default=2),The starting value of p, the order (or number of time lags) of the auto-regressive (“AR”) model. Must be a positive integer.
        :input start_q                     : int, optional (default=2) The starting value of q, the order of the moving-average (“MA”) model. Must be a positive integer.
        :input d                           : int, optional (default=None) The order of first-differencing.None (by default)
        :input max_p                       : int, optional (default=5) The maximum value of p, inclusive. Must be a positive integer greater than or equal to start_p.
        :input max_d                       : int, optional (default=2) The maximum value of d, or the maximum number of non-seasonal differences. Must be a positive integer greater than or equal to d.
        :input max_q                       : int, optional (default=5) The maximum value of q, inclusive. Must be a positive integer greater than start_q.
        :input start_P                     : int, optional (default=1) The starting value of P, the order of the auto-regressive portion of the seasonal model.
        :input D                           : int, optional (default=None) The order of the seasonal differencing. If None (by default, the value will automatically be selected based on the results of the seasonal_test. Must be a positive integer or None.
        :input start_Q                     : int, optional (default=1) The starting value of Q, the order of the moving-average portion of the seasonal model.
        :input max_P                       : int, optional (default=2) The maximum value of P, inclusive. Must be a positive integer greater than start_P.
        :input max_D                       : int, optional (default=1) The maximum value of D. Must be a positive integer greater than D.
        :input max_Q                       : int, optional (default=2) The maximum value of Q, inclusive. Must be a positive integer greater than start_Q.
        :input m                           : int, optional (default=1) The period for seasonal differencing, m refers to the number of periods in each season. For example, m is 4 for quarterly data, 12 for monthly data, or 1 for annual (non-seasonal) data. Default is 1. Note that if m == 1 (i.e., is non-seasonal), seasonal will be set to False.
        :input seasonal                    : bool, optional (default=True) Whether to fit a seasonal ARIMA. Default is True. Note that if seasonal is True and m == 1, seasonal will be set to False.
        :input test                        : str, optional (default=’kpss’) Type of unit root test to use in order to detect stationarity if stationary is False and d is None.
        :input seasonal_test               : str, optional (default=’ocsb’) This determines which seasonal unit root test is used if seasonal is True and D is None.
        :input stepwise                    : bool, optional (default=True) Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters. The stepwise algorithm can be significantly faster than fitting all (or a random subset of) hyper-parameter combinations and is less likely to over-fit the model.
        :input suppress_warnings           : bool, optional (default=False) Many warnings might be thrown inside of statsmodels. If suppress_warnings is True, all of the warnings coming from ARIMA will be squelched.
        :input error_action                : str, optional (default=’warn’) If unable to fit an ARIMA due to stationarity issues, whether to warn (‘warn’), raise the ValueError (‘raise’) or ignore (‘ignore’). Note that the default behavior is to warn, and fits that fail will be returned as None.
        :input trace                       : bool or int, optional (default=False) Whether to print status on the fits. A value of False will print no debugging information. A value of True will print some. Integer values exceeding 1 will print increasing amounts of debug information at each fit.
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        df['train'] = df['train'].replace(0,0.001)
        
        model = pm.auto_arima(df["train"][:-m],start_p=start_p,d=d,start_q=start_q,max_p=max_p,max_d=max_d,max_q=max_q,start_P=start_P,D=D,start_Q=start_Q,max_P=max_P,max_D=max_D,max_Q=max_Q,m=m,seasonal=seasonal,test=test,seasonal_test=seasonal_test,trace=trace,error_action=error_action, suppress_warnings=suppress_warnings, stepwise=stepwise)
        fit1 = model.fit(df["train"][:-m])
        params = fit1.params
        df[column_name] = np.nan
        #y_fit = fit1.fittedvalues
        y_fit = fit1.predict_in_sample()
        y_fore = fit1.predict(m)
        
        
        df[column_name][:-m] = y_fit
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
        
        start_p=[1]
        d = [1]
        start_q=[1]
        max_p=[2]
        max_d=[2]
        max_q=[2]
        start_P=[1]
        D=[1] 
        start_Q=[1]
        max_P=[2]
        max_D=[2]
        max_Q=[2]
        m=[53]
        #global small_d
        #small_d=[1]
        
        seasonal=[True]
        test=['adf']
        seasonal_test=['ocsb', 'ch']
        trace = [True]
        error_action = ['ignore']
        suppress_warnings = [True]
        stepwise = [True]
        #t_params = ['n','c','t','ct']
        #enforce_stationarity=[False]
        #enforce_invertibility=[False]
        
        #print(start_q)
        
        hyperparams = [(startp,small_d,startq,maxp,maxd,maxq,startP,big_D,\
                        startQ,maxP,maxD,maxQ,small_m,si,test_d,seasonal_test_D,tr,er_ac, supp_war, step_wise) \
                       for startp in start_p for small_d in d for startq in start_q \
                       for maxp in max_p for maxd in max_d for maxq in max_q \
                       for startP in start_P for big_D in D for startQ in start_Q for maxP in max_P for maxD in max_D \
                       for maxQ in max_Q for small_m in m for si in seasonal for test_d in test for seasonal_test_D in seasonal_test \
                       for tr in trace for er_ac in error_action for supp_war in suppress_warnings \
                       for step_wise in stepwise]
                       #for ti in t_params]
        
        #hyperparams = [(False, None, None, None, ti, si, None) if not asi else (asi, sli, ssi, ssei, ti, si, dsi) for (asi, sli, ssi, ssei, ti, si, dsi) in hyperparams]
        
        hyperparams = list(set(hyperparams))

        
        hyperparam_combinations = [{'start_p':startp, 'd':small_d,'start_q':startq,'max_p':maxp,'max_d':maxd,\
                                    'max_q':maxq,'start_P':startP,'D':big_D,'start_Q':startQ,'max_P':maxP,'max_D':maxD,\
                                    'max_Q':maxQ,'m':small_m,'seasonal':si,'test':test_d,'seasonal_test':seasonal_test_D,\
                                    'trace':tr,'error_action':er_ac,'suppress_warnings':supp_war,\
                                    'stepwise':step_wise}
                                    #,'trend':ti} \
                                    for (startp,small_d,startq,maxp,maxd,maxq,startP,big_D,startQ,maxP,maxD,maxQ,small_m,si,test_d,seasonal_test_D,tr,er_ac, supp_war, step_wise) in hyperparams]
        
        print(f'hyperparams: {hyperparams}')
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