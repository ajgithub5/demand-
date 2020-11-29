# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 08:14:24 2020

@author: p3000445
"""

from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np

from models.Model import Model

class Weekly_Trend(Model):
    def __init__(self, model_name):
        super().__init__(model_name)
        
    def model(self, column_name, df, wgt0=None, wgt1=None, wgt2=None):
    
        """
        performs predictions using the weekly-trend model approach
        
        :input column_name           : str, name of column to hold the predicted values
        :input df                    : dataframe, weekly-level data
        :input prediction_period     : int, default=8, the number of weeks to predict ahead for
    
        :returns df_pred             : dataframe, weekly-level, with predictions
        :returns params              : dictionary, default=None, placeholder for saving the best hyperparameters chosen by the model, if not provided as arguments to this method
 
        """
        
        
        prediction_period = self.prediction_period
        df_train = df.copy()[:-prediction_period]
        agg_value = {}
        agg_value1 ={}
        #-------------------------------Create weekly trend columns based on prediction_period
        for i in np.arange(1,prediction_period+1,1):
            colname="Weekly trend "+str(i)
            agg_value = "Weekly trend "+str(i), 'mean'
            agg_value1.update([(agg_value)])
           
            df_train[colname]=0.0
        
        
        for i in np.arange(1,len(df_train['train']),1):
            for j in np.arange(1,prediction_period+1,1):
                colname="Weekly trend "+str(j)
                if i>=j:
                    if df_train['train'][i-j]!=0:
                        df_train[colname][i] = np.clip((df_train['train'][i]/df_train['train'][i-j]).round(2),a_min=None,a_max=2)
                    else:
                        df_train[colname][i] = 0
                df_train[colname].replace(0, np.nan, inplace=True)
                
        #====================== Taking weighted mean ================================
        
#        print("Starting with weighted mean calculations")
#        
#        Year_wt = pd.DataFrame(sorted(list(set(df_train["Year"])),reverse = True),columns = ['Year'])
#        Year_wt['Weights'] = 0.1
#        
#        Year_wt['Weights'].iloc[0] = wgt0
#        Year_wt['Weights'].iloc[1] = wgt1
#        Year_wt['Weights'].iloc[2] = wgt2
#        
#        df_train = pd.merge(df_train,Year_wt,on='Year',how='inner')
#        df_train['Weights'] = df_train['Weights']/((len(Year_wt['Year']) - 3)*0.1 + wgt0 + wgt1 +wgt2)
#        
#        agg_value1 ={}
#        for i in np.arange(1,prediction_period+1,1):
#            colname="Weekly trend "+str(i)
#            df_train[colname] = df_train[colname]*df_train['Weights']
#            agg_value = "Weekly trend "+str(i), 'sum'
#            agg_value1.update([(agg_value)])       
                
                
        
        #=======================Take mean of weekly trend================================
        
        df_mean_weekly = df_train.groupby(['BusWeekNum'], as_index = False).agg(agg_value1)
    
    
        #------------------------Predictions---------------------------------
        
        df_pred = pd.DataFrame()
        #df_pred = df_test[0:prediction_period+1]
        df_pred = df[-prediction_period-1:]
        #df_pred = df_pred.loc[:,'Year':'test']
        
        df_pred=pd.merge(df_pred,df_mean_weekly,how='inner',on=['BusWeekNum'], suffixes=('', '_y'))
        
        df_pred[column_name]=0
        #df_pred['Iteration']=1
        for i in np.arange(1, len(df_pred['train']),1):
            colname="Weekly trend "+str(i)
            df_pred[column_name][i] = (df_pred['train'][0]*df_pred[colname][i])
          
        #df_pred[column_name] = df['train']
        df_pred = df_pred[-prediction_period:]    
        df_train[column_name] = df_train['train']
        df_pred = pd.concat([df_train, df_pred[0:]])
        
        df_pred = df_pred[[col for col in df_pred.columns if 'Weekly trend' not in col]]
        return df_pred
    
    def cv_search(self, df_splits, column_name, logic, df_daily, team_location, hyperparam_combinations=None):
        
#        wgt0_range = np.arange(0.1,0.8,0.1)
#        wgt1_range = np.arange(0.1,0.6,0.1)
#        wgt2_range = np.arange(0.1,0.3,0.1)
#
#        #wgt0_range = [0.1]
#        #wgt1_range = [0.1]
#        #wgt2_range = [0.1]
#        
#        hyperparams = [(wgt0,wgt1,wgt2) for wgt0 in wgt0_range for wgt1 in wgt1_range for wgt2 in wgt2_range]
#        hyperparam_combinations = [{'wgt0':wgt0,'wgt1':wgt1,'wgt2':wgt2} for (wgt0,wgt1,wgt2) in hyperparams]
#        
        #if non-parametric model, hyperparam_combinations = None
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