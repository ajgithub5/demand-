# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from utils.training_helper import model_metrics, create_framework, get_n_splits, accuracy_metrics
from data.preprocessing import shorter_week_deflation, deflation_logic
import pandas as pd
import numpy as np
import math
from abc import ABC, abstractmethod
import json
import traceback

class Model(ABC):
    
    def __init__(self, model_name):
        with open('..\\config\\config.json', 'r') as file:
            config = json.load(file)
        
        self.model_name = model_name
        self.prediction_period = config['prediction_period'] if config['run_mode']=='forecast'\
                                                             else config['evaluation_period']
        self.wf_step = 1
        print(f'Initializing model: {self.model_name}')
        print(f'Prediction period set to {self.prediction_period}')
        print(f'Walk-forward step-size set to {self.wf_step}')
    
    @abstractmethod
    def model(self):
        pass
    
    @abstractmethod
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
        
        # do not perform gridsearch for non-parametric models
        if not hyperparam_combinations:
            return None
        
        m = self.prediction_period
        wf = self.wf_step
        accuracy_j = []
        
        #hyperparam_combinations = hyperparam_combinations[0:1]
        #print('Returning First Hyperparameter Combination')
        #return hyperparam_combinations[0]
    
        for hyperparam_combination in hyperparam_combinations:
            accuracy_i = []
            for id, dfi in enumerate(df_splits):
                dfo = self.model(column_name, dfi, **hyperparam_combination)
                #dfo = shorter_week_deflation(column_name,dfo)
                dfo = deflation_logic(logic, column_name, dfo, df_daily, team_location)
                accuracy_i.append(accuracy_metrics(dfo["test"][-m:-m+wf], dfo[column_name][-m:-m+wf]))
            accuracy_j.append((np.mean([x[0] for x in accuracy_i]),hyperparam_combination))
        #optimal_hyperparams = min(accuracy_j)[1]
        scores = [j[0] for j in accuracy_j]
        optimal_hyperparams = accuracy_j[scores.index(min(scores))][1]
    
        print(f'optimal_hyp: {optimal_hyperparams}')
        return optimal_hyperparams

        
    
    @abstractmethod
    def evaluate(self, df, column_name, logic, team_name, df_daily, team_location):
    
        """
        evluates model metrics on all walk-forward split dataframes using the predictions using the most optimal hyperparameters chosen from grid-search method
    
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
        with open('..\\config\\config.json', 'r') as file: config = json.load(file)
        num_iterations = config['num_iterations']
        
        try:
            m  = self.prediction_period
            wf = self.wf_step
            model = self.model_name
            
            cv_results = []
            df_main_framework = pd.DataFrame()
            df_splits = get_n_splits(math.ceil(53/wf), df)[0:num_iterations ]
            
            optimal_hyperparams = self.cv_search(df_splits, column_name, logic, df_daily, team_location)
            
            '''
            if optimal_hyperparams.get('status')=='failed':
                return pd.DataFrame(), {}, optimal_hyperparams
            '''
            print(f'optimal hyperparameters chosen: {optimal_hyperparams}')
            
            for id, dfi in enumerate(df_splits):
                print('Evaluating {} on split# {}'.format(self.model_name,id))
                dfo = self.model(column_name, dfi, **optimal_hyperparams) if optimal_hyperparams else self.model(column_name, dfi)    # using best hyperparam
                #dfo = shorter_week_deflation(column_name,dfo) 
                dfo = deflation_logic(logic, column_name, dfo, df_daily, team_location)
                cv_results.append(accuracy_metrics(dfo["test"][-m:-m+wf], dfo[column_name][-m:-m+wf])) # append cv_results for every iteration
                #df_framework = dfo[-m:]
                df_framework = dfo[-m:]
                df_framework["Iteration"] = id
                df_framework = create_framework(df_framework, column_name, logic, model)
                df_main_framework = pd.concat([df_main_framework,df_framework])
            
            framework = df_main_framework
            mape = np.mean([x[0] for x in cv_results])
            rmse = np.mean([x[1] for x in cv_results])
            pwt = np.mean([x[2] for x in cv_results])
            
            
            metrics = model_metrics(team_name, logic, model, mape, rmse, pwt)

            return {
                    'status':True, 
                    'content':(framework, metrics, optimal_hyperparams)
                    }
        
        except Exception as e:
            return {
                    'status':False,
                    'error':traceback.format_exc()
                    }