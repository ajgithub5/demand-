# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 07:26:06 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
import pandas as pd
import numpy as np
#
from models.Model import Model

class SES(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name, df,apply_smoothing,smoothing_level = None):
    
        """
        performs predictions using the simple exponential smoothing model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input apply_smoothing       : bool, indicates whether to factor-in smoothing parameters in the Holt model
        :input smoothing_level       : int, default=None, l parameter in Simple Exponential Smoothing model
        
        
        :returns df                  : dataframe, weekly-level, with predictions
        """
   
        m = self.prediction_period
        
        if apply_smoothing == True:
            fit1 = SimpleExpSmoothing(df["train"][:-m]).fit(smoothing_level=smoothing_level,optimized=True)
            params = None
        elif apply_smoothing == False:
            fit1 = SimpleExpSmoothing(df["train"][:-m]).fit(optimized = True)
            params = fit1.params
        y_fit = fit1.fittedvalues
        y_fore = fit1.forecast(m)
        df[column_name]=np.nan
        #df[column_name][:-1] = y_fit
        df[column_name][:-m] = df['train'].iloc[:-m]
        df[column_name][-m:] = y_fore
        
        #df[column_name].iloc[-1:] = list(y_pred)[-1]
        return df
    
    def ADE_base_model(self, column_name, df,apply_smoothing,smoothing_level = None):
    
        """
        performs predictions using the simple exponential smoothing model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input apply_smoothing       : bool, indicates whether to factor-in smoothing parameters in the Holt model
        :input smoothing_level       : int, default=None, l parameter in Simple Exponential Smoothing model
        
        
        :returns df                  : dataframe, weekly-level, with predictions
        """
        
        m = self.prediction_period
        
        if apply_smoothing == True:
            fit1 = SimpleExpSmoothing(df["train"][:-m]).fit(smoothing_level=smoothing_level,optimized=True)
            params = None
        elif apply_smoothing == False:
            fit1 = SimpleExpSmoothing(df["train"][:-m]).fit(optimized = True)
            params = fit1.params
        y_fit = fit1.fittedvalues
        y_fore = fit1.forecast(m)
        print(f' len of fittedvalues: {len(y_fit)}')
        df[column_name]=np.nan
        #df[column_name][:-1] = y_fit
        df[column_name][:-m] = y_fit
        df[column_name][-m:] = y_fore
        
        df[column_name] = df[column_name].ffill().bfill()
        return df
    
    def cv_search(self, df_splits,column_name, logic, df_daily, team_location, hyperparam_combinations=None):
    
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
        '''
        apply_smoothing = [True, False]
        smoothing_level = np.arange(0,1,0.1)
        '''
        apply_smoothing = [False]
        smoothing_level = [None]       
        
        hyperparams = [(asi, sli) for asi in apply_smoothing for sli in smoothing_level]
        hyperparams = [(asi, None) if not asi else (asi, sli) for (asi, sli) in hyperparams]
        hyperparams = list(set(hyperparams))
        print(f'hyperparams: {hyperparams}')
        
        #hyperparams = get_random_grid(hyperparams, random_seed=0, sample_size=0.25)
        #hyperparams.append((False,None))
        #print(f'random hyperparams: {hyperparams}')
        hyperparam_combinations = [{'apply_smoothing':asi, 'smoothing_level':sli} for (asi, sli) in hyperparams]
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