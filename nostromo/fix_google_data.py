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
    tot_del=0
    indexList=list()
    for name, group in g:
        if 1==1:
            badRows=group.query('high <=0 |  low <=0 | open <=0 | close <=0 | volume <0')
            if (len(badRows)==1):
                #group.loc[group.open <=0,'open']=group['open'].mean()
                group.loc[group.high <=0,'high']=group['high'].mean()
                #group.loc[group.low <=0,'low']=group['low'].mean()
                #group.loc[group.close <=0,'close']=group['close'].mean()
                #group.loc[group.volume <=0,'volume']=group['volume'].mean()
                
                
            elif (len(badRows)>1):
                print 'found ',len(badRows)
                maxDate=badRows['date'].max(axis=0)
                rowsToDelete=group[group.date <= maxDate]
                len_all=len(group)
                len_todel=len(rowsToDelete)
                print '[',name,'] will delete',len_todel, 'out of ',len_all
                if len(rowsToDelete.index) > 0:
                    indexList.append(rowsToDelete.index)
                tot_del+=len_todel
                
    flat_list = [item for sublist in indexList for item in sublist]
    v = np.asarray(flat_list)
    a=v.flatten()
    return a

theTypes={'open'    : np.float32,'high'    : np.float32, 'low'    : np.float32, 'close'    : np.float32, 'volume'    : np.float32, 
'adj_open': np.float32,'adj_high': np.float32, 'adj_low': np.float32, 'adj_close': np.float32, 'adj_volume': np.float32 }

df=pd.read_csv('../data/stocks/2007-2017_stocks.csv',parse_dates=True, header=0,dtype=theTypes)
df.sort_values(by=['ticker','date'],ascending=[True,True],inplace=True)
print len(df)
a=cleanup(df)
print len(a)
print len(df)
print a[0:10]
df.drop(a,inplace=True)
print len(df)
#df=df.round(2)
#newdf=df.copy(deep=True)
df.to_csv('../data/stocks/stocks_fixed.csv',float_format='%.4f',index=True)