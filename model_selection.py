# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 06:39:01 2020

@author: a08473
"""

import pandas as pd
import numpy as np



def model_selector(df):
    df['BEST_MODEL']=0
    while True:
        df1 = df[df['BEST_MODEL']==1]
        df0 = df[df['BEST_MODEL']!=1]
        df0 = sort_models(df0)
        df = pd.concat([df1,df0],ignore_index=True)
        if 0 not in df['BEST_MODEL'].tolist():
            break
        
    for i in range(1,df.shape[0]):
        df['BEST_MODEL'].iloc[i]=i+1
    
    return df
def sort_models(df):
    

    df.reset_index(drop=True, inplace=True)
    thresholdmape = (min(df['Cross_Validated_MAPE']))*1.1
    df1 = df[df['Cross_Validated_MAPE']<=thresholdmape]
    thresholdpwt = (max(df1['Cross_Validated_PWT']))*0.9
    df2 = df1[df1['Cross_Validated_PWT']>=thresholdpwt]
    df3 = df2[df2['Cross_Validated_RMSE']==min(df2['Cross_Validated_RMSE'])]
    df3['BEST_MODEL']=1
    df4 = df.loc[list(set(df.index)-set(df3.index))]
    df5 = pd.concat([df3, df4])
    
    return df5




      
    
            
    