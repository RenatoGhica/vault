#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:17:06 2018

@author: developer
"""

import pandas as pd
import numpy as np
import os
import sys

theTypes={'open'    : np.float32,'high'    : np.float32, 'low'    : np.float32, 'close'    : np.float32, 'volume'    : np.float32, 
'adj_open': np.float32,'adj_high': np.float32, 'adj_low': np.float32, 'adj_close': np.float32, 'adj_volume': np.float32 }

path='../data/Stocks'


final_df = pd.DataFrame(columns=['ticker','date','open', 'high', 'low', 'close', 'volume','openint'])
all_dfs=list()
for file in sorted(os.listdir(path)):
    current = os.path.join(path, file)
    if os.path.isfile(current):
        #print current
        
        f = os.path.basename(current)        
        s = os.path.getsize(current)
        if (s > 0):
            #print 'processing :',f
            ticker= f.split('.')[0].upper()
            df=pd.read_csv(current,parse_dates=True, header=0,dtype=theTypes)
            df.insert(loc=0,column='ticker',value=ticker)
            df.rename(columns={'Date': 'date','Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume','OpenInt': 'openint',  },inplace=True)
            df.sort_values(by=['ticker','date'],ascending=[True,True],inplace=True)
            out_f='../data/results/'+ticker+'.csv'
            #df.to_csv(out_f,float_format='%.4f',index=False)
            all_dfs.append(df)
            #final_df=pd.concat([final_df,df])
            print f,':processing :',ticker, '(',len(df),'records',')'

print 'concatenating...'
final_df=pd.concat(all_dfs)
print 'done!'
final_df.sort_values(by=['ticker','date'],ascending=[True,True],inplace=True)
final_df.to_csv('../data/results/STOCKS.csv',float_format='%.4f',index=False)        

