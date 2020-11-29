# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 05:56:43 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np
from models.Model import Model

class SMA(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name, df, nweeks):
        """
        performs predictions using the simple moving average model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input nweeks                : int, number of weeks to consider for rolling mean

        
        :returns df                  : dataframe, weekly-level, with predictions
        
        """
        m = self.prediction_period
        n = nweeks
        df[column_name] = df["train"].rolling(window=n).mean()
        df[column_name] = df[column_name].shift(1)
        pred = df[-m:][column_name].iloc[0]
        df[column_name].iloc[:-m] = df['train'].iloc[:-m]
        df[column_name].iloc[-m:] = pred
        
        
        return df
    
    def ADE_base_model(self, column_name, df, nweeks):
        """
        performs predictions using the simple moving average model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input nweeks                : int, number of weeks to consider for rolling mean

        
        :returns df                  : dataframe, weekly-level, with predictions
        
        """
        m = self.prediction_period
        n = nweeks
        df[column_name] = df["train"].rolling(window=n).mean()
        df[column_name] = df[column_name].shift(1)
        pred = df[-m:][column_name].iloc[0]
        df[column_name].iloc[-m:] = pred

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
       
        
        nweeks = np.arange(2,53,1)
        hyperparams = [(n,) for n in nweeks]

        hyperparam_combinations = [{'nweeks':n} for (n,) in hyperparams]
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
        result = super().evaluate(df, column_name, logic, team_name, df_daily, team_location)
        
        if result['status']:
            return result
        else:
            return {
                    'status':False,
                    'error':result['error']
                    }
