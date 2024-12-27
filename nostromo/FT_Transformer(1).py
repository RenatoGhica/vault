import pandas as pd
import numpy as np
import datetime as dt
import talib as ta
import itertools 
import re, math



from talib import MA_Type
from pandas.tseries.offsets import BDay

import holidays
from pandas.tseries.holiday import USFederalHolidayCalendar
from datetime import datetime
from scipy.ndimage.interpolation import shift
from scipy.fftpack import fft
from FT_DataFactory import FT_DataFactory
from enum import Enum
from datetime import date
from scipy import stats
import keras.utils

from itertools import groupby

from sklearn.preprocessing import MinMaxScaler



class colType(Enum):
    BOOLEAN='.boolean'
    PCT='.pct'
    SCALAR='.scalar'
    UNITLESS='.unitless'
    

#ONE_DAY = dt.timedelta(days=1)
#HOLIDAYS_US = holidays.US()

class FT_Transformer:
    
# =============================================================================
#     
# =============================================================================
    def makeFeatureName(self,l_name,fn,r_name,fnMiddle=False):
        feature_str=''
        l_name=str(l_name)
        r_name=(str(r_name).zfill(3)) if (len(str(r_name)) > 0) else ''
        
        if (l_name.endswith(feature_str) or r_name.endswith(feature_str)):
            l_name=re.sub(feature_str, '', l_name)
            r_name=re.sub(feature_str, '', r_name)
            
        #result='('+l_name+'_'+fn+'_'+str(r_name).zfill(0)+')'+feature_str
        if (fnMiddle):
            result='('+l_name + ' '+fn+' '+r_name + ')'+feature_str
        else:
            result=fn + '.'+l_name + '.'+r_name + feature_str
        
        return result

# =============================================================================
# 
# =============================================================================
    def applyUnary(self,theDataFrame, theFunction, theFunctionName,theList):
        df=theDataFrame 
        for x,y in theList:
             colName=x+' '+theFunctionName+' '+y
             df[colName]=theFunction(df[x],df[y])
# =============================================================================
# 
# =============================================================================
    def _ndh(self,theDate):
        next_day = theDate + self._1_DAY
        if (next_day in self._HC):
            results=False 
        else:
            results=True
        return results	
	
# =============================================================================
#     
# =============================================================================
        
    def _nbd(self,theDate):
        next_day = theDate + self._1_DAY
        while next_day.weekday() in self._WKND or next_day in self._HC:
            next_day += self._1_DAY
        return next_day

# =============================================================================
# get previous N business days
# =============================================================================
    def _prev_bday(self,theDate,n=30):
        results=list()
        prev_day = theDate 
        for i in range(0,n):
            #prev_day = theDate - self._1_DAY
            while prev_day.weekday() in self._WKND or prev_day in self._HC:
                prev_day -= self._1_DAY
                theDate=prev_day
            results.append(prev_day)
            prev_day -= self._1_DAY
        results=sorted(results)
        return results
# =============================================================================
# 
# =============================================================================
    def _next_bday(self,theDate,n=5):
        results=list()
        next_day = theDate 
        next_day += self._1_DAY
        for i in range(0,n):
            #prev_day = theDate - self._1_DAY
            while next_day.weekday() in self._WKND or next_day in self._HC:
                next_day += self._1_DAY
                theDate=next_day
            results.append(next_day)
            next_day += self._1_DAY
        results=sorted(results)
        return results

# =============================================================================
# 
# =============================================================================
    def myRSI (self,theSeries,thePeriod=14):
        
        delta = np.diff(theSeries)
        #delta=delta[1:]
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
    
        # Calculate the EWMA
        roll_up1 = pd.stats.moments.ewma(up, thePeriod)
        pd.rolling_mean(up,thePeriod)
        roll_down1 = pd.stats.moments.ewma(np.abs(down), thePeriod)
        
        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        
        # Calculate the SMA
        roll_up2 = pd.rolling_mean(up, thePeriod)
        roll_down2 = pd.rolling_mean(np.abs(down), thePeriod)
        
        # Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))    
        
        RSI1=np.insert(RSI1,0,0)
        RSI2=np.insert(RSI2,0,0)
        return RSI1
    



    def myZ(self,x,t):
        r=self.rolling_window(x,t)
        results = (x[t:] - np.mean(r))/np.std(r)
        h=np.full([t],np.nan)
        results = np.concatenate((h,results))
        return results
        
    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def strength(self,x,y,t):
        cutoff=0.95
        empty=np.full((t),np.nan)
        results = np.concatenate((empty,x[t:]/np.amax(self.rolling_window(y,t),axis=1)[:-1]))
        results[results < cutoff]=0
        return results
    
    def weakness(self,x,y,t):
        cutoff=0.0
        empty=np.full((t),np.nan)
        results = np.concatenate((empty,(np.amin(self.rolling_window(y,t),axis=1)[:-1])/x[t:]))
        #results = np.concatenate((empty,np.amin(self.rolling_window(y,t),axis=1)[:-1]))
        #results = np.concatenate((empty,x[t:]/np.amin(self.rolling_window(y,t),axis=1)[:-1]))
        #results[results < cutoff]=0
        return results

    
    def normalizeMinMax(self,x,ax=0):

        results = (x - np.nanmin(x,axis=ax)) / (np.nanmax(x,axis=ax)-np.nanmin(x,axis=ax))
        #results = (x - np.nanmax(x,axis=ax) + (x - np.nanmin(x,axis=ax) )) / (np.nanmax(x,axis=ax)-np.nanmin(x,axis=ax))
        results = results * 2 - 1
        return results
    
    def normalize(self,x):
        #results = (x - np.nanmean(x)) / (np.nanstd(x))
        results = (x - np.nanmean(x)) / (np.nanvar(x))
        #results = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        #results = results * 2 - 1
        return results

    def bucketize(self,x,theBuckets):
        return (np.digitize(x,theBuckets))
        
    def ema(self,x,t,p=30):
        v = (x/ shift(x,t))-1
        v[v==np.inf]=0
        results = ta.EMA(v,p)            
        return results
        
    def customTarget(self,x,t):
        a = ta.LINEARREG_ANGLE (x,t)
        results = (a > 0)
        return results

    def myRSI2(self,x,t):
        results = ta.RSI(x,t)/100. 
        results[results > 1]=1
        results[results < 0]=0
        return results
        
# =============================================================================
# 
# =============================================================================
    def __init__(self,theCalendar=holidays.US()):
        #self.itsReturnBuckets=np.linspace(-1,1,3)
        self.itsReturnBuckets=np.linspace(0,0,1)
        
        #self.itsCustomBuckets=np.linspace(0,50,51)
        self.itsCustomBuckets=np.linspace(1,5,5)
        #self.itsCustomBuckets=np.linspace(1,5,5)
        #self.itsReturnBuckets=np.linspace(-0.0001,0.0001,3)
        self.itsZScoreBuckets=np.linspace(-3,3,11)
        print self.itsReturnBuckets
        print self.itsZScoreBuckets
        print self.itsCustomBuckets
        #self.itsReturnBuckets=np.linspace(-3.0,3.0,15)
        
        pd.options.mode.chained_assignment = None  # default='warn'

        self._HC=theCalendar # holiday calendar
        self._1_DAY=dt.timedelta(days=1)        
        
        self._WK_MON=0
        self._WK_TUE=1
        self._WK_WED=2
        self._WK_THU=3
        self._WK_FRI=4
        self._WK_SAT=5
        self._WK_SUN=6
        
        self._WKND=holidays.WEEKEND
        
        self._MO_EOQ=[3,6,9,12]
        self._MO_EARNINGS=[1,4,7,10]
        
        
    
        
        #self.itsTimeSlices=[2,3,5,8,13,21]
        #self.itsTimeSlicesGroup2=[2,3,5,8,13,21]
        #self.itsTimeSlices=[5,20,30,50,100]
        
        
        
        
        #self.itsTimeSlices=[5,14,20,30,40,50,60,70,80,90,100]
        self.itsTimeSlices=[5,20,50,100]
        #self.itsTimeSlices=[3,5,8,13,21,34,55]
        self.itsTimeSlicesGroup2=[5,20,50,100]
        
        
        
        self.TAGroup_1= [
                #[['#rand'],['high'],     self.itsTimeSlices,colType.UNITLESS],
                #[['#rsi'],['high','low','close'],     self.itsTimeSlices,colType.UNITLESS],
                [['#rsi'],['high','low'],     self.itsTimeSlices,colType.UNITLESS],
                [['#ema'],['high','low'],     self.itsTimeSlices,colType.UNITLESS],
                
                [['#lr_slope'],['high','low'],     self.itsTimeSlices,colType.UNITLESS],
                [['#lr_angle'],['high','low'],     self.itsTimeSlices,colType.UNITLESS],
                #[['#rocp'],['high'],         [2,3,4,5],colType.UNITLESS],
                
                #[['#zscore'],['high'],     self.itsTimeSlices,colType.UNITLESS],
                [['#raw_value'],['high','low'],    [0],colType.UNITLESS],   
                  

                #[['#ema'],['close',], self.itsTimeSlices,colType.SCALAR],
                #[['#std'],['high','low'], self.itsTimeSlices,colType.UNITLESS],
                #[['#max'],['high','low'], self.itsTimeSlices,colType.UNITLESS],
                #[['#min'],['high','low'], self.itsTimeSlices,colType.UNITLESS],
                
                
                [['#return'],['high'],    [5],colType.PCT],                
                [['#return_category'],['high'],    [5],colType.UNITLESS],   
                
                  #[['#custom_target'],['high'],    [0],colType.UNITLESS],   

                
                #[['#isHoliday'],        ['date'],[0],colType.BOOLEAN],
                #[['#isFriday'],         ['date'],[0],colType.BOOLEAN],
                #[['#isEom'],            ['date'],[0],colType.BOOLEAN],
                #[['#isEoq'],            ['date'],[0],colType.BOOLEAN],
                #[['#isEarningsSeason'], ['date'],[0],colType.BOOLEAN],
                #[['#isBeforeHol'],      ['date'],[0],colType.BOOLEAN],
                #[['#isOptExp'],         ['date'],[0],colType.BOOLEAN],
                #[['#isOptExp_4X'],      ['date'],[0],colType.BOOLEAN],
                #[['std'],['volume'],[30,60]]
                
        ]
        
        self.TAGroup_2= [
                #[['#mfi'],['olhc'],     self.itsTimeSlices,colType.PCT],
                #[['#adx'],['olhc'],     self.itsTimeSlices,colType.PCT],

                #[['#adx','#+di','#-di','#+dm','#-dm'],['ohlc'], self.itsTimeSlices,colType.PCT],
                [['#mfi'],['ohlc'], self.itsTimeSlices,colType.UNITLESS],
                #[['#bop'],['ohlc'],[0],colType.UNITLESS], 
                #[['#gu'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                #[['#gd'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                #[['#hh'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                #[['#ll'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                #[['#ch'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                #[['#cl'],['ohlc'],self.itsTimeSlicesGroup2,colType.BOOLEAN],    
                


        ]

        # take existing columns and compare all combinations    
        self.TAGroup_3 = [
                #[['#rsi'],['.div_ab.','.div_ba.'],colType.UNITLESS],
                #[['#ema'],['.div_ab.','.div_ba.'],colType.UNITLESS],
                #[['#rsi'],['>','<'],colType.BOOLEAN],
                #[['#ema'],['>','<'],colType.BOOLEAN],
                #[['#std'],['>','<'],colType.BOOLEAN],
                #[['#min'],['>','<'],colType.BOOLEAN],
                #[['#max'],['>','<'],colType.BOOLEAN],
                ]
        
        
        self.functions= { 
                '#rand'                 : (lambda x,t: np.random.rand(len(x))),                
                '#lr_slope'             : (lambda x,t: ta.LINEARREG_SLOPE (x,t)),
                '#lr_angle'             : (lambda x,t: ta.LINEARREG_ANGLE (x,t)),
                '#rocp'                 : (lambda x,t: ta.ROCP (x,t)),
                '#rsi'                  : (lambda x,t: ta.RSI(x,t)/100.),
                '#rsi__'                : (lambda x,t: self.myRSI2(x,t)),
                '#ema'                  : (lambda x,t: ta.EMA(x,t)),
                '#ema_'                  : (lambda x,t: self.normalizeMinMax(ta.EMA(x,t))),
                '#std'                  : (lambda x,t: ta.STDDEV(x,t)),
                '#min'                  : (lambda x,t: ta.MIN(x,t)),                  #np.sort(x[-t:])[0])
                '#max'                  : (lambda x,t: ta.MAX(x,t)),                  

                '#raw_value'            : (lambda x,t: x),               
                #'#return'               : (lambda x,t: (x / shift(x,t))-1),               
                '#return'               : (lambda x,t: ta.ROCP (x,t)),               
                #'#return_category'      : (lambda x,t: self.bucketize((  (x / shift(x,t)) -1),self.itsReturnBuckets)),  
                '#return_category'      : (lambda x,t: self.customTarget(x,t)) ,  
                #'#return_category'      : (lambda x,t: self.getCategories(x,t)),  
                #'#return_ma'            : (lambda x,t: self.ema(x,t,20)  ),               
                #'#return_ma_category'   : (lambda x,t: self.bucketize(self.ema(x,t,20),self.itsReturnBuckets)),  
                
                '#custom_target'        : (lambda x,t: self.bucketize(self.customTarget(x,t),self.itsCustomBuckets)),  
                
                
                
                '#zscore'               : (lambda x,t: ((x - ta.EMA(x,t))/ ta.STDDEV(x,t)) ),
                '#zscore_category'      : (lambda x,t: self.bucketize((x - ta.EMA(x,t))/ ta.STDDEV(x,t),self.itsZScoreBuckets)),                
                #'>'                     : (lambda x,y : (1 if (x > y) else -1)),
                #'<'                     : (lambda x,y : (1 if (x < y) else -1)),

                '.div_ab.'              : (lambda x,y : x/y),
                '.div_ba.'              : (lambda x,y : y/x),
                '>'                     : (lambda x,y : (x > y).astype(int)),
                '<'                     : (lambda x,y : (x < y).astype(int)),
                
                '#adx'                  : (lambda d,o,h,l,c,v,t : ta.ADX(h,l,c,t)/100.),
                '#+di'                  : (lambda d,o,h,l,c,v,t : ta.PLUS_DI(h,l,c,t)/100.),
                '#-di'                  : (lambda d,o,h,l,c,v,t : ta.MINUS_DI(h,l,c,t)/100.),
                '#+dm'                  : (lambda d,o,h,l,c,v,t : ta.PLUS_DM(h,l,t)/100.),
                '#-dm'                  : (lambda d,o,h,l,c,v,t : ta.MINUS_DM(h,l,t)/100.),
                '#mfi'                  : (lambda d,o,h,l,c,v,t : ta.MFI(h,l,c,np.float64(v),t)/100.),
                '#bop'                  : (lambda d,o,h,l,c,v,t : ta.MFI(o,h,l,c)),
                
                '#gu'                   : (lambda d,o,h,l,c,v,t : self.strength(o,h,t)),
                '#gd'                   : (lambda d,o,h,l,c,v,t : self.weakness(o,l,t)),

                '#hh'                   : (lambda d,o,h,l,c,v,t : self.strength(h,h,t)),
                '#ll'                   : (lambda d,o,h,l,c,v,t : self.weakness(l,l,t)),

                '#ch'                   : (lambda d,o,h,l,c,v,t : self.strength(c,l,t)),
                '#cl'                   : (lambda d,o,h,l,c,v,t : self.weakness(c,h,t)),

# =============================================================================
#                 '#gu'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_up,0,o,y=h,n=t)),
#                 '#ch'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_up,0,c,y=h,n=t)),
#                 '#hh'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_up,0,h,y=h,n=t)),
# 
#                 '#gd'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_down,0,o,y=l,n=t)),
#                 '#cl'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_down,0,c,y=l,n=t)),
#                 '#ll'                   : (lambda d,o,h,l,c,v,t : np.apply_along_axis(self.gap_down,0,h,y=l,n=t)),
# 
# =============================================================================
                
                '#isHoliday'            : (lambda theDates,t :[ int((pd.to_datetime(d) in self._HC)) for d in theDates] ),
                '#isFriday'             : (lambda theDates,t :[ int((pd.to_datetime(d).weekday()==self._WK_FRI )) for d in theDates] ),
                '#isEom'                : (lambda theDates,t :[ int((self._nbd(pd.to_datetime(d)).month != pd.to_datetime(d).month)) for d in theDates] ),
                '#isEoq'                : (lambda theDates,t :[ int((self._nbd(pd.to_datetime(d)).month != pd.to_datetime(d).month) and pd.to_datetime(d).month in self._MO_EOQ) for d in theDates] ),
                '#isEarningsSeason'     : (lambda theDates,t :[ int(pd.to_datetime(d).month in self._MO_EARNINGS) for d in theDates] ),
                '#isBeforeHol'          : (lambda theDates,t :[ int((((self._nbd(pd.to_datetime(d)).weekday() - pd.to_datetime(d).weekday())) > 1) or (abs(((self._nbd(pd.to_datetime(d)).weekday() - pd.to_datetime(d).weekday()))) == 3)) for d in theDates] ),
                '#isOptExp'             : (lambda theDates,t :[ int((pd.to_datetime(d).weekday()== self._WK_FRI and (14< pd.to_datetime(d).day < 22) and  pd.to_datetime(d) not in self._HC) or 
                                                                    (pd.to_datetime(d).weekday()== self._WK_THU and (14< pd.to_datetime(d).day < 22) and  pd.to_datetime(d) + self._1_DAY in self._HC)
                                                                    ) for d in theDates] ),
                
                '#isOptExp_4X'          : (lambda theDates,t :[ int((pd.to_datetime(d).month in (self._MO_EOQ) and pd.to_datetime(d).weekday()==4 and (14< pd.to_datetime(d).day < 22) and  pd.to_datetime(d) not in self._HC) or 
                                                                    (pd.to_datetime(d).month in (self._MO_EOQ) and pd.to_datetime(d).weekday()==3 and (14< pd.to_datetime(d).day < 22) and  pd.to_datetime(d) + self._1_DAY in self._HC)
                                                                    ) for d in theDates] ),
                
                
    
            }
        

 

# =============================================================================
# 
# =============================================================================
    def xNormalize(self,theDataFrame,outFile='../data/modeldata_stage1.csv',theTicker='ticker',theDate='date'):
        MIN_TIMESAMPLES=30
        
        df=theDataFrame
        #T0=self.itsTransformsDate       
        theGroups=list()
        
        g=df.groupby(by=[theTicker])
        for name, group in g:
            print 'processing :<',name,'>, with ',len(group),' items'
            if len(group) < MIN_TIMESAMPLES:
                print 'skipping [',name,'] - series too short! (',len(group),' < ',MIN_TIMESAMPLES,')'
                continue

        
# =============================================================================
# 
# =============================================================================
    def xform(self,theDataFrame,outFile='../data/modeldata_stage1.csv',theTicker='ticker',theDate='date'):
        MIN_TIMESAMPLES=30
        
        df=theDataFrame
        #T0=self.itsTransformsDate       
        theGroups=list()
        
        print '0.1:',datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        g=df.groupby(by=[theTicker])
        print '0.2:',datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
        for name, group in g:
            print 'processing :<',name,'>, with ',len(group),' items'
            if len(group) < MIN_TIMESAMPLES:
                print 'skipping [',name,'] - series too short! (',len(group),' < ',MIN_TIMESAMPLES,')'
                continue
            group.sort_values(by=[theDate])
            d=group['date']
            o=group['open']
            h=group['high']
            l=group['low']
            c=group['close']
            v=group['volume']
            
            colsGroup1=list()
            colsGroup2=list()
            colsGroup3=list()

            #rsi,ema,std            
            for theFunction,theNames, thePeriods,theToken in self.TAGroup_1:
                theList= list(itertools.product(theFunction,theNames, thePeriods)) 
                for fn_name,col,t in theList:
                    colName=self.makeFeatureName(col,fn_name,t)
                    group[colName]=self.functions.get(fn_name)(group[col].values,t)
                    colsGroup1.append([colName,colName+theToken.value])
                    
            #adx
            for theFunction,theNames, thePeriods,theToken in self.TAGroup_2:
                theList= list(itertools.product(theFunction,theNames, thePeriods))
                for fn_name,col,t in theList:
                    colName=self.makeFeatureName(col,fn_name,t)
                    group[colName]=self.functions.get(fn_name)(d.values,o.values,h.values,l.values,c.values,v.values,t)
                    colsGroup2.append([colName,colName+theToken.value])
            
            theGroup_3_results=dict()
            
            for theFunction,theNames,theToken in self.TAGroup_3:
                theList= list(itertools.product(theFunction,theNames,))
                #print theList
                for pattern, fn_name in theList:
                    #print pattern,fn_name
                    theFunction=self.functions.get(fn_name)
                    df1=group.loc[:,group.columns.str.contains(pattern)]
                    #print group.columns
                    l= list(itertools.combinations(df1.columns,2))
                   
                    for col_left,col_right in l:
                        colName=self.makeFeatureName(col_left,fn_name,col_right,True)
                        theGroup_3_results[colName]=theFunction(group[col_left],group[col_right])
                        colsGroup3.append([colName,colName+theToken.value])
            for key,value in theGroup_3_results.iteritems():
                    group[key]=value
                    #print key
                                      
                    
            theGroups.append(group)
            
        results=pd.concat(theGroups)

                    

        
        
        
        cols1=['ticker','date','open','high','low','close','volume'] 
        cols2=sorted(group[group.columns.difference(cols1)])
        cols=cols1+cols2

        results=results.reindex_axis(cols,axis=1)
        colsGroup=dict(dict(colsGroup1).items() + dict(colsGroup2).items() + dict(colsGroup3).items())
        
        results.rename(columns=dict(colsGroup),inplace=True)

        results.head()   
        results=results.round(4)
        
        results.to_csv(outFile)
        #return results

# =============================================================================
#     Sanity check
# =============================================================================

def test():
        pass
# =============================================================================
#     f_stage1    ='../data/modeldata_stage1.csv'
#     f_stage2    ='../data/modeldata_stage2.csv'
#     f_stats     ='../data/modeldata_mystats.csv'
#     
#     m=FT_Model()
#     m.normalize(inFile=f_stage1,outFile=f_stage2)
#     m.featureSummary(inFile=f_stage2,outFile=f_stats)
#     r1=m.target('#return.close.001.pct')
# 
# =============================================================================

if __name__ == "__main__":
    test()               
    
    
    