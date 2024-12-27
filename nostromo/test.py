#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:17:06 2018

@author: developer
"""

import pandas as pd
import numpy as np

def cleanup(df):
    g = df.groupby(by=['ticker'])
    df_len=len(df)
    tot_del=0
    indexList=list()
    for name, group in g:
        if 1==1:
            #badRows=group[group.high <=0 |  group.low <=0 | group.open <=0 | group.close <=0 | group.volume <0]        
            badRows=group.query('high <=0 |  low <=0 | open <=0 | close <=0 | volume <0')
            #print group
            #print badRows
            if (len(badRows)>1):
                maxDate=badRows['date'].max(axis=0)
                rowsToDelete=group[group.date <= maxDate]
                len_all=len(group)
                len_todel=len(rowsToDelete)
                print '[',name,'] will delete',len_todel, 'out of ',len_all
                indexList.append(rowsToDelete.index)
                tot_del+=len_todel
                
    print 'deleting ',tot_del,' out of ',df_len, ':',tot_del/df_len * 1.0
    #v=np.asarray(indexList)
    flat_list = [item for sublist in indexList for item in sublist]
    v = np.asarray(flat_list)
    print v.shape
    print v[0]
    
    a=v.flatten()
    print a.shape
    df.drop(a, inplace=True)
    print 'df before : ',df_len,' after:',len(df)

theTypes={'open'    : np.float32,'high'    : np.float32, 'low'    : np.float32, 'close'    : np.float32, 'volume'    : np.float32, 
'adj_open': np.float32,'adj_high': np.float32, 'adj_low': np.float32, 'adj_close': np.float32, 'adj_volume': np.float32 }

df=pd.read_csv('../data/2007-2017_stocks.csv',parse_dates=True, header=0,dtype=theTypes)
df.sort_values(by=['ticker','date'],ascending=[True,True],inplace=True)
df.info()
cleanup(df)
print len(df)
df.to_csv('../data/stocks/stocks_fixed.csv')