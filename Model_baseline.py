# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 07:34:31 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression

from models.Model import Model

class Baseline(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
        
    def apply_weights(self, row, df, max_weight=1, min_weight=0):
    
        """
        calculates weight for a given year, to be used for multiplying with that year's seasonality#
        
        :input row        : row of dataframe
        :input df         : weekly-level dataframe
        :input max_weight : int, default=1. The weight to be applied to the most recent year in the dataframe
        :input min_weight : int, default=0. The weight to be applied to the most earliest year in the dataframe
        
        :returns          : weight to be applied to given year of that row
        
        """
        max_year = max(df['Year'])
        min_year = min(df['Year'])
        step_size = (max_weight - min_weight)/(df['Year'].unique().shape[0]-1)
        if row.loc['Year'] == max_year:
            return max_weight
        elif row.loc['Year'] == min_year:
            return min_weight
        else:
            return min_weight + step_size * (row.loc['Year']-min_year)
    
    def model(self, column_name,df, model_web="multiplicative", weighted_seasonality = False, max_weight=1, 
              seasonal_periods = 53, trend_history = 4):
    
        """
        performs predictions using the baseline model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input model_web             : str, 'multiplicative' or 'additive', used for statsmodels seasonal_decompose method
        :input weighted_seasonality  : bool, to indicate whether to apply more or equal weightage to recent years as compared to earlier years' seasonality components
        :input max_weight            : int, default=1, to indicate max weight to apply to recent year's seasonality component, if weighted_seasonality=True
        :input freq                  : int, default=53, used as an input to statsmodels seasonal_decompose method
        :input trend_history         : int, default=4, indicates how many years back from the most recent year to consider for training the Linear Regression model to predict the trend component
    
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        print('seasonal periods: {}'.format(seasonal_periods))
        m = self.prediction_period
        df_main = df.copy(deep=True)
        df=df[:-m]
        
        # save a copy of original data frame for slicing it per trend history and doing linear regression training on trend component of it 
        df_trend = df.copy(deep=True)
        
        result = seasonal_decompose(df["train"].values, model=model_web, freq=seasonal_periods)
        df["observed"]= result.observed
        df["trend"] = result.trend
        df["resid"] = result.resid
        df["seasonal"] = result.seasonal
        #weighted_seasonality = True
        if weighted_seasonality == False:
            df['Weight']=1
        else:
            df['Weight']=df.apply(self.apply_weights,df=df, min_weight=0, max_weight=max_weight, axis=1)
        df['Weighted Seasonality']=(df['Weight']*df['seasonal'])
        
        Average_seasonality = (df.groupby('BusWeekNum')['Weighted Seasonality'].sum()) / df.groupby('BusWeekNum')['Weight'].sum()
        
        max_year = max(df_trend['Year'].tolist())
        start_year = max_year - trend_history
        start_index = min(df_trend[df_trend['Year']==start_year].index.values)
        df_trend = df_trend.loc[start_index: ]
        df = df_trend
        result = seasonal_decompose(df["train"].values, model=model_web, freq=seasonal_periods)
        df["observed"]= result.observed
        df["trend"] = result.trend
        df["resid"] = result.resid
        df["seasonal"] = result.seasonal
        '''
        print(f' df size before null removal: {df.shape}')
        df = df[~df["trend"].isnull()]
        print(f' df size after null removal: {df.shape}')
        '''
        #df['trend']=df['trend'].replace(np.nan,0)
        #df['trend'].replace(np.Inf,0)
        #df['trend'].replace(-np.Inf,0)
        df = df[~df["trend"].isnull()]
        
        X = df.index.values.reshape(-1,1)
        y = pd.DataFrame(df["trend"])
        X = pd.DataFrame(X, columns=["X"])
        #X["trend"] = y["trend"].tolist()
        model = LinearRegression()
    
        fit = model.fit(X,y)
        
        df = df_main
        timesteps = df_main.index.values.reshape(-1,1)
        df[column_name] = model.predict(timesteps)
        df[column_name][:-m] = df['train'].iloc[:-m]
        indices = Average_seasonality.index.tolist()
        values = Average_seasonality.values.tolist()
        d = dict(zip(indices,values))
        df["avg_seasonality"] = df["BusWeekNum"]
        #df.replace({"avg_seasonality": d}, inplace=True)
        df["avg_seasonality"] = df["avg_seasonality"].map(d)
        if model_web == 'multiplicative':
            df[column_name] = df[column_name]*df["avg_seasonality"]
        else:
            df[column_name] = df[column_name]+df["avg_seasonality"]
        df = df.reset_index()
        
        #df = shorter_week_deflation(column_name,df)
        #df = deflation_logic(logic, column_name, df)
        return df
    
    def ADE_base_model(self, column_name,df, model_web="multiplicative", weighted_seasonality = False, max_weight=1, freq = 53, trend_history = 4):
    
        """
        performs predictions using the baseline model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input model_web             : str, 'multiplicative' or 'additive', used for statsmodels seasonal_decompose method
        :input weighted_seasonality  : bool, to indicate whether to apply more or equal weightage to recent years as compared to earlier years' seasonality components
        :input max_weight            : int, default=1, to indicate max weight to apply to recent year's seasonality component, if weighted_seasonality=True
        :input freq                  : int, default=53, used as an input to statsmodels seasonal_decompose method
        :input trend_history         : int, default=4, indicates how many years back from the most recent year to consider for training the Linear Regression model to predict the trend component
    
        :returns df                  : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method

        """
        #1/0
        
        m = self.prediction_period
        df_main = df.copy(deep=True)
        df=df[:-m]
        
        # save a copy of original data frame for slicing it per trend history and doing linear regression training on trend component of it 
        df_trend = df.copy(deep=True)
        
        result = seasonal_decompose(df["train"].values, model=model_web, freq=53)
        df["observed"]= result.observed
        df["trend"] = result.trend
        df["resid"] = result.resid
        df["seasonal"] = result.seasonal
        #weighted_seasonality = True
        if weighted_seasonality == False:
            df['Weight']=1
        else:
            df['Weight']=df.apply(self.apply_weights,df=df, min_weight=0, max_weight=max_weight, axis=1)
        df['Weighted Seasonality']=(df['Weight']*df['seasonal'])
        
        Average_seasonality = (df.groupby('BusWeekNum')['Weighted Seasonality'].sum()) / df.groupby('BusWeekNum')['Weight'].sum()
        
        max_year = max(df_trend['Year'].tolist())
        start_year = max_year - trend_history
        start_index = min(df_trend[df_trend['Year']==start_year].index.values)
        df_trend = df_trend.loc[start_index: ]
        df = df_trend
        result = seasonal_decompose(df["train"].values, model=model_web, freq=53)
        df["observed"]= result.observed
        df["trend"] = result.trend
        df["resid"] = result.resid
        df["seasonal"] = result.seasonal
        '''
        print(f' df size before null removal: {df.shape}')
        df = df[~df["trend"].isnull()]
        print(f' df size after null removal: {df.shape}')
        '''
        #df['trend']=df['trend'].replace(np.nan,0)
        #df['trend'].replace(np.Inf,0)
        #df['trend'].replace(-np.Inf,0)
        df = df[~df["trend"].isnull()]
        
        X = df.index.values.reshape(-1,1)
        y = pd.DataFrame(df["trend"])
        X = pd.DataFrame(X, columns=["X"])
        #X["trend"] = y["trend"].tolist()
        model = LinearRegression()
    
        fit = model.fit(X,y)
        
        df = df_main
        timesteps = df_main.index.values.reshape(-1,1)
        df[column_name] = model.predict(timesteps)
        #df[column_name][:-m] = df['train'].iloc[:-m]
        indices = Average_seasonality.index.tolist()
        values = Average_seasonality.values.tolist()
        d = dict(zip(indices,values))
        df["avg_seasonality"] = df["BusWeekNum"]
        #print(f' dictionary {d}')
        df.replace({"avg_seasonality": d}, inplace=True)
        if model_web == 'multiplicative':
            df[column_name] = df[column_name]*df["avg_seasonality"]
        else:
            df[column_name] = df[column_name]+df["avg_seasonality"]
        df = df.reset_index()
        
        #df = shorter_week_deflation(column_name,df)
        #df = deflation_logic(logic, column_name, df)
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

        
        model_web = ['multiplicative','additive']
        trend_history = [1,2]
        weighted_seasonality = [False, True]
        max_weight = np.arange(1,5,1)
        '''
        model_web = ['multiplicative', 'additive']
        trend_history = [1]
        weighted_seasonality = [False]
        max_weight = [None]
        '''
        hyperparams = [(mwi, thi, wsi, mxwi) for mwi in model_web for thi in trend_history for wsi in weighted_seasonality for mxwi in max_weight]
        hyperparams = [(mwi, thi, False, None) if not weighted_seasonality else (mwi, thi, wsi, mxwi) for (mwi, thi, wsi, mxwi) in hyperparams]
        #hyperparams = list(set(hyperparams))[-1:]
        #hyperparams = get_random_grid(hyperparams, random_seed=0, sample_size=0.25)
        #print(f'random hyperparams: {hyperparams}')
        
        hyperparam_combinations = [{'model_web':mwi, 'trend_history':thi, 'weighted_seasonality':wsi, 'max_weight':mxwi} for (mwi, thi, wsi, mxwi) in hyperparams]

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