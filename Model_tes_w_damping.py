# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 06:46:50 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np
#from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from models.Model import Model

class TES_w_damping(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name,df, apply_smoothing, trend, seasonal, smoothing_level=None,smoothing_slope=None,smoothing_seasonal=None
              , damping_slope=None, seasonal_periods=53):
    
        """
        performs predictions using the triple exponential smoothing with damping model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input apply_smoothing       : bool, indicates whether to factor-in smoothing parameters in the Holt model
        :input trend                 : str, 'additive' or 'multiplicative', indicates the type of model w.r.t trend component 
        :input seasonal              : str, 'additive' or 'multiplicative', indicates the type of model w.r.t seasonal component 
        
        :input smoothing_level       : int, default=None, l parameter in Holt-Winter's model
        :input smoothing_slope       : int, default=None, b parameter in Holt-Winter's model
        :input smoothing_seasonal    : int, default=None, s parameter in Holt-Winter's model
        :input damping_slope         : int, default=None, phi parameter in Holt-Winter's model
        :input seasonal_periods      : int, default=53, parameter used by statsmodels to calculate the seasonality component
        
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        #df['train'] = df['train'].replace(0,0.001)
        if apply_smoothing == True:
            fit1 = ExponentialSmoothing(df["train"][:-m], 
                                        trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(smoothing_level = smoothing_level, 
                                                          smoothing_slope = smoothing_slope,
                                                          smoothing_seasonal = smoothing_seasonal,
                                                          damping_slope = damping_slope,
                                                          optimized = True)
            params = None
        elif apply_smoothing == False:
            fit1 = ExponentialSmoothing(df["train"][:-m],  trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(optimized = True)
            params = fit1.params
            if np.isnan(params['initial_slope']):
                print('Initial Slope is Undefined')
                fit1 = ExponentialSmoothing(df["train"][:-m],  trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(damping_slope=0.1, optimized = True)
                params = fit1.params
                print('Model is refitted with damping slope fixed at 0.1')
            print('====================')
            print(params)
            print('====================')
        df[column_name] = np.nan
        #y_fit = fit1.fittedvalues
        y_fore = fit1.forecast(m)
        #y_fore = fit1.predict(df.shape[0]-m)
        df[column_name][:-m] = df['train'].iloc[:-m]
        #df[column_name][:-1] = y_fit
        df[column_name][-m:] = y_fore

    
        return df
    
    def ADE_base_model(self, column_name,df, apply_smoothing, trend, seasonal, smoothing_level=None,smoothing_slope=None,smoothing_seasonal=None, damping_slope=None, seasonal_periods=53):
    
        """
        performs predictions using the triple exponential smoothing with damping model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input apply_smoothing       : bool, indicates whether to factor-in smoothing parameters in the Holt model
        :input trend                 : str, 'additive' or 'multiplicative', indicates the type of model w.r.t trend component 
        :input seasonal              : str, 'additive' or 'multiplicative', indicates the type of model w.r.t seasonal component 
        
        :input smoothing_level       : int, default=None, l parameter in Holt-Winter's model
        :input smoothing_slope       : int, default=None, b parameter in Holt-Winter's model
        :input smoothing_seasonal    : int, default=None, s parameter in Holt-Winter's model
        :input damping_slope         : int, default=None, phi parameter in Holt-Winter's model
        :input seasonal_periods      : int, default=53, parameter used by statsmodels to calculate the seasonality component
        
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        
        m = self.prediction_period
        
        #df['train'] = df['train'].replace(0,0.001)
        if apply_smoothing == True:
            fit1 = ExponentialSmoothing(df["train"][:-m], 
                                        trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(smoothing_level = smoothing_level, 
                                                          smoothing_slope = smoothing_slope,
                                                          smoothing_seasonal = smoothing_seasonal,
                                                          damping_slope = damping_slope,
                                                          optimized = True)
            params = None
        elif apply_smoothing == False:
            fit1 = ExponentialSmoothing(df["train"][:-m],  trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(optimized = True)
            params = fit1.params
            if np.isnan(params['initial_slope']):
                print('Initial Slope is Undefined')
                fit1 = ExponentialSmoothing(df["train"][:-m],  trend = trend, 
                                        seasonal=seasonal,
                                        seasonal_periods=seasonal_periods,
                                        damped=True).fit(damping_slope=0.1, optimized = True)
                params = fit1.params
                print('Model is refitted with damping slope fixed at 0.1')
            print('====================')
            print(params)
            print('====================')
        df[column_name] = np.nan
        y_fit = fit1.fittedvalues
        y_fore = fit1.forecast(m)
        
        #df[column_name][:-m] = df['train'].iloc[:-m]
        
        df[column_name][:-m] = y_fit
        df[column_name][-m:] = y_fore

    
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
        
#        apply_smoothing = [True, False]
#        smoothing_levels = np.arange(0,1,0.1)
#        smoothing_slopes = np.arange(0,1,0.1)
#        smoothing_seasonals = np.arange(0,1,0.1)
#        trend = ['multiplicative', 'additive']
#        seasonal = ['multiplicative', 'additive']
#        damping_slopes = np.arange(0,1,0.1)
        
        #return {'apply_smoothing': False, 'smoothing_level': None, 'smoothing_slope': None, 'smoothing_seasonal': None, 'trend': 'multiplicative', 'seasonal': 'multiplicative', 'damping_slope': None}
        #return {'apply_smoothing': False, 'smoothing_level': None, 'smoothing_slope': None, 'smoothing_seasonal': None, 'trend': 'multiplicative', 'seasonal': 'multiplicative', 'damping_slope': None}
        
        apply_smoothing = [False]
        smoothing_levels = [None]
        smoothing_slopes = [None]
        smoothing_seasonals = [None]
        trend = ['multiplicative', 'additive']
        seasonal = ['multiplicative', 'additive']
        damping_slopes = [None]
        
        hyperparams = [(asi, sli, ssi, ssei, ti, si, dsi) for asi in apply_smoothing for sli in smoothing_levels for ssi in smoothing_slopes for ssei in smoothing_seasonals for ti in trend for si in seasonal for dsi in damping_slopes]
        hyperparams = [(False, None, None, None, ti, si, None) if not asi else (asi, sli, ssi, ssei, ti, si, dsi) for (asi, sli, ssi, ssei, ti, si, dsi) in hyperparams]
        hyperparams = list(set(hyperparams))
#        hyperparams = get_random_grid(hyperparams, random_seed=0, sample_size=0.25)
#        hyperparams.append((False,None,None,None,'multiplicative','multiplicative',None))
#        hyperparams.append((False,None,None,None,'multiplicative','additive',None))
#        hyperparams.append((False,None,None,None,'additive','multiplicative',None))
#        hyperparams.append((False,None,None,None,'additive','additive',None))
#        print(f'random hyperparams: {hyperparams}')
        
        hyperparam_combinations = [{'apply_smoothing':asi, 'smoothing_level':sli, 'smoothing_slope':ssi, 'smoothing_seasonal': ssei, 'trend':ti, 'seasonal':si, 'damping_slope':dsi} for (asi, sli, ssi, ssei, ti, si, dsi) in hyperparams]
        print(f'hyperparams: {hyperparams}')
        optimal_hyperparams = super().cv_search(df_splits, column_name, logic, df_daily, team_location, hyperparam_combinations)
        
        return optimal_hyperparams
        

    #Main wrapper for tes_w_damping model train (gridsearch) and test on cv splits
    
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