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
import json
from abc import ABC, abstractmethod
import traceback
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from models.Model import Model
from models.model_object_picker import get_model_object



class ADE(Model):
    
    def __init__(self, model_name, base_models):
        super().__init__(model_name)
        self.base_models = base_models
        
        self.top_n = 4
        self.omega = 8
        self.K = 8
        
    def apply_softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        '''
        x = np.array(x)
        return np.exp(-x) / np.sum(np.exp(-x), axis=0)
        '''
        S = sum(x)
        n = len(x)-1
        weights = [(S-e)/(n*S) for e in x]
        return np.asarray(weights)
    
    
    def model(self, column_name, input_dfs, df_daily, team_location, base_models):
        
        m  = self.prediction_period
        wf = self.wf_step
        #base_models = self.base_models
        
        print(base_models)
        
        base_model_names         = [base_model['model_name'] for base_model in base_models]        
        base_learners            = [get_model_object(base_model['model_name']) for base_model in base_models]
        base_optimal_hyperparams = [base_model['optimal_hyperparameters'] for base_model in base_models]
        base_logics              = [base_model['logic'] for base_model in base_models]
        base_outputs=[]
        
        params = {'max_depth': 5, 'min_samples_split': 2, 
                  'n_estimators': 1000, 'subsample':0.8, 'learning_rate': 0.1,  
                  'loss': 'ls', 'random_state':0}
        #meta_learners = [RandomForestRegressor(max_depth=2, random_state=0) for i in range(len(base_learners))]
        meta_learners = [GradientBoostingRegressor(**params) for i in range(len(base_learners))]
        base_mapes = []
        
        # Filter the top_n base-learners (based on MAPE/RMSE) base learners and their meta learner counterparts
        #dfi = df.copy()
        exceptions = []
        for bl,bo,l,ml in zip(base_learners,base_optimal_hyperparams,base_logics,meta_learners):
            try:
                dfi = input_dfs[l].copy()
                dfo = bl.ADE_base_model(column_name, dfi, **bo) if bo else bl.ADE_base_model(column_name, dfi)    # using best hyperparam     
                dfo = shorter_week_deflation(column_name,dfo)
                dfo = deflation_logic(l, column_name, dfo, df_daily, team_location)
                #base_mapes.append(accuracy_metrics(dfo['test'][-m:-m+wf],dfo[column_name][-m:-m+wf])[0])
                #base_mapes.append(accuracy_metrics(dfo['test'][-m-wf:-m],dfo[column_name][-m-wf:-m])[0])
                #base_mapes.append((dfo[column_name][-m-self.omega:-m]-dfo['test'][-m-self.omega:-m])/dfo['test'][-m-self.omega:-m])
                base_mapes.append(accuracy_metrics(dfo['test'][-m-self.omega:-m],dfo[column_name][-m-self.omega:-m])[3])
            
            except Exception as e:
                base_mapes.append(10e10)
                exceptions.append('Error in base model {}: {}'.format(bl,traceback.format_exc()))
                pass
        
        positive_mpes = sorted([m for m in base_mapes if m>=0])
        negative_mpes = sorted([m for m in base_mapes if m<0])
        positive_mpes = positive_mpes[0:self.top_n//2]
        negative_mpes = negative_mpes[0:self.top_n//2]
        positive_indices = [base_mapes.index(p) for p in positive_mpes]
        negative_indices = [base_mapes.index(n) for n in negative_mpes]
        all_indices = positive_indices + negative_indices
        
        sorted_base_logics = [base_logics[i] for i in all_indices]
        sorted_base_model_names = [base_model_names[i] for i in all_indices]
        sorted_base_learners = [base_learners[i] for i in all_indices]
        sorted_base_optimal_hyperparams = [base_optimal_hyperparams[i] for i in all_indices]
        sorted_meta_learners = [meta_learners[i] for i in all_indices]
        
        print('====')
        print(sorted_base_logics)
        print(sorted_base_model_names)
        print(sorted_base_learners)
        print(sorted_base_optimal_hyperparams)
        print(sorted_meta_learners)
        print('====')
        
        num_base_models = len(all_indices)
        
        '''
        sorted_items  = sorted(zip(base_mapes, base_logics, base_model_names, base_learners, base_optimal_hyperparams, meta_learners))
        sorted_base_mapes, sorted_base_logics, sorted_base_model_names, sorted_base_learners,sorted_base_optimal_hyperparams, sorted_meta_learners = map(list, zip(*sorted_items))
        
        sorted_base_model_names = sorted_base_model_names[0:self.top_n]
        sorted_base_learners = sorted_base_learners[0:self.top_n]
        sorted_base_optimal_hyperparams = sorted_base_optimal_hyperparams[0:self.top_n]
        sorted_base_logics = sorted_base_logics[0:self.top_n]
        sorted_meta_learners = sorted_meta_learners[0:self.top_n]
        '''
        
        # fit the meta learners on Error¬Actuals and predict the Error for the last m rows of data
        for l,dfi in zip(input_dfs.keys(), input_dfs.values()):
            for k in range(1,self.K+1):
                    dfi['test_{}'.format(k)] = np.nan
                    dfi['test_{}'.format(k)] = dfi['test'].shift(k)
            input_dfs[l]=dfi.copy()
        
        #dfi = df.copy()
        for bl,bo,l,ml in zip(sorted_base_learners, sorted_base_optimal_hyperparams, sorted_base_logics, sorted_meta_learners):
            dfi = input_dfs[l].copy()
            dfo = bl.ADE_base_model(column_name, dfi, **bo) if bo else bl.ADE_base_model(column_name, dfi)    # using best hyperparam     
            dfo = shorter_week_deflation(column_name,dfo)
            dfo = deflation_logic(l, column_name, dfo, df_daily, team_location)
            dfo['Meta_Predictions'] = np.nan
            '''
            ml.fit(
                    np.asarray(dfo[['test_{}'.format(k) for k in range(1,self.K+1)]][self.K:-m]),
                    np.asarray(np.abs((dfo[column_name][self.K:-m]-dfo['test'][self.K:-m])/dfo['test'][self.K:-m]))
                    )
            '''
            ml.fit(
                    np.asarray(dfo[['test_{}'.format(k) for k in range(1,self.K+1)]][self.K:-m]),
                    np.asarray((dfo[column_name][self.K:-m]-dfo['test'][self.K:-m])/dfo['test'][self.K:-m])
                    )
            pred = ml.predict(np.asarray(dfo[['test_{}'.format(k) for k in range(1,self.K+1)]][-m:-m+wf]))[0]
            
            dfo['Meta_Predictions'][self.K:-m] = ml.predict(np.asarray(dfo[['test_{}'.format(k) for k in range(1,self.K+1)]][self.K:-m]))
            dfo['Meta_Predictions'][-m:] = pred
            base_outputs.append(dfo.copy())
            

        
        #df_ADE = df.copy()
        df_ADE = input_dfs[list(input_dfs)[0]].copy()
        df_ADE[column_name]=np.nan
        df_ADE['Base Models']=np.nan
        df_ADE['Base Models']=str(sorted_base_model_names)
        for i,df_b in enumerate(base_outputs):
            df_ADE[str(column_name)+'_{}'.format(i)] = df_b[column_name]
            df_ADE['Meta_Predictions'+'_{}'.format(i)] = df_b['Meta_Predictions']
            df_ADE['Softmax_Weights'+'_{}'.format(i)] = np.nan
            
        for j in range(m):
            softmax_weights = self.apply_softmax(df_ADE.iloc[-m+j][['Meta_Predictions'+'_{}'.format(i) for i in range(num_base_models)]].tolist())
            for i in range(num_base_models):
                #df_ADE.iloc[-m+j]['Error_Weights'+'_{}'.format(i)] = softmax_weights[i]
                df_ADE.set_value(df_ADE.shape[0]-m+j,'Softmax_Weights'+'_{}'.format(i),softmax_weights[i])
            #df_ADE.iloc[-m+j][column_name]
            df_ADE.set_value(df_ADE.shape[0]-m+j,
                             column_name,
                             np.sum(
                                    [
                                    df_ADE.iloc[-m+j]['Softmax_Weights'+'_{}'.format(i)]
                                    *df_ADE.iloc[-m+j][[str(column_name)+'_{}'.format(i)]]
                                    for i in range(num_base_models)
                                    ])
                            )
    
        
        return {
                'df':df_ADE,
                'exceptions':exceptions
                }
    
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
        
        # do not perform gridsearch for non-parametric models
        return None

    
    
    def evaluate(self, input_dfs, column_name, team_name, df_daily, team_location):
    
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
        '''
        framework, metrics, _ = super().evaluate(df, column_name, logic, team_name, df_daily, team_location)
        return framework, metrics, {'ADE':self.base_models}
        '''
        
        with open('..\\config\\config.json', 'r') as file: config = json.load(file)
        num_iterations = config['num_iterations']
        
        m  = self.prediction_period
        wf = self.wf_step
        model_name = self.model_name

        base_models = self.base_models
        
        
        cv_results = []
        df_main_framework = pd.DataFrame()
        
        df_splits = {li:get_n_splits(math.ceil(53/wf), dfi)[0:num_iterations] for li, dfi in zip(input_dfs.keys(),input_dfs.values())}

        exceptions = []
        for id in range(math.ceil(53/wf))[0:num_iterations]:
            try:
                #if id==0 or id==1:
                #    0/0
                input_dfs = {li:df_splits[li][id] for li in input_dfs.keys()}
                print(input_dfs)
                result = self.model(column_name, input_dfs, df_daily, team_location, base_models)
                df_ADE = result['df']
                exceptions.append('Error in iteration {}: {}'.format(id, result['exceptions']))
                cv_results.append(accuracy_metrics(df_ADE["test"][-m:-m+wf], df_ADE[column_name][-m:-m+wf])) # append cv_results for every iteration
                #df_framework = df_ADE[-m:]
                df_framework = df_ADE.copy()
                df_framework["Iteration"] = id
                df_framework = create_framework(df_framework, column_name, None, model_name)
                df_main_framework = pd.concat([df_main_framework,df_framework])
            except Exception as e:
                exceptions.append('Error in iteration {}: {}'.format(id, traceback.format_exc()))
                continue
        if len(cv_results) > 0:
            framework = df_main_framework
            mape = np.mean([x[0] for x in cv_results])
            rmse = np.mean([x[1] for x in cv_results])
            pwt = np.mean([x[2] for x in cv_results])
            metrics = model_metrics(team_name, None, model_name, mape, rmse, pwt)
            return {
                    'status':True,
                    'content':(framework, metrics, {'base_models':self.base_models}),
                    'exceptions':exceptions
                    }
        else:
            return {
                    'status':False,
                    'error':exceptions
                    }
        