# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 07:46:10 2020

@author: p3000445
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 07:40:07 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np

from models.Model import Model

class WMA60(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
    
    def model(self, column_name, df, nweeks):
        
        
    
        """
        performs predictions using the weighted moving average model (new approach - considering only last 60 weeks in calculating the rolling mean)
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input n                     : int, number of weeks to consider for rolling mean
        
        :returns new_df              : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        m = self.prediction_period
        n = nweeks
        
        weights = []
        for x in range(df.shape[0])[1:61]:
            weights.append(x)
        weights = [x/sum(weights) for x in weights]
        new_df = df[-(60+m):]
        
        new_df["weights"] = np.nan
        new_df["weights"].iloc[:60] = weights
        new_df["weighted_train"] = new_df["train"]*new_df["weights"] 
        #new_df[column_name] = new_df["weighted_train"].rolling(window=n).mean()
        new_df[column_name] = new_df["weighted_train"].rolling(window=n).sum()/new_df["weights"].rolling(window=n).sum()
        new_df[column_name] = new_df[column_name].shift(1)
        #new_df[column_name] = new_df[column_name]/sum(weights[-n:])
        pred = new_df[-m:][column_name].iloc[0]
        new_df[column_name].iloc[:-m] = new_df['train'].iloc[:-m]
        new_df[column_name].iloc[-m:] = pred
        
        df[column_name] = np.nan
        df[column_name].iloc[-(60+m):] = new_df[column_name]
        df[column_name].iloc[:-(60+m)] = df['train'].iloc[:-(60+m)]
        
        #print(df)
        return df
    
    def cv_search(self, df_splits,column_name, logic, df_daily, team_location, hyperparams={}):
    
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
        
        
        nweeks = np.arange(2,61,1)
        hyperparams = [(n,) for n in nweeks]
        print(f'hyperparams: {hyperparams}')
        
        #hyperparams = get_random_grid(hyperparams, random_seed=0, sample_size=0.25)
        #print(f'random hyperparams: {hyperparams}')
        hyperparam_combinations = [{'nweeks':n} for (n,) in hyperparams]
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