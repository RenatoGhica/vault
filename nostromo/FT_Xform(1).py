fimport pandas as pd
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

import logging
            
# =============================================================================
# HELPERS
# =============================================================================
def _getOHLCV(self,g):
    o=g['open']
    h=g['high']
    l=g['low']
    c=g['close']
    v=g['volume']
    return o,h,l,c,v
# =============================================================================
# 
# =============================================================================
class colType(Enum):
    BOOLEAN='.boolean'
    PCT='.pct'
    SCALAR='.scalar'
    UNITLESS='.unitless'
# =============================================================================
#     
# =============================================================================
class FT_Xform:

# =============================================================================
# 
# =============================================================================
    def __init__(self,theDataframe):
        self.itsDataframe=theDataframe
        self.itsTickerName='ticker'

        logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(levelname)-8s] -> %(message)s',)
        self.itsLogger = logging.getLogger('FT_Xform')

       
        self.Technicals_1= [
                [['#rsi'],['high','close'],     self.itsTimeSlices, colType.UNITLESS],
                [['#raw_value'],['high'],       [0],                colType.UNITLESS],   
                [['#return'],['high'],          [5],                colType.PCT],                
                [['#return_category'],['high'], [5],                colType.UNITLESS]
                 ]

        self.functions= { 
                '#rand'                 : (lambda x,t: np.random.rand(len(x))),                
                '#rsi'                  : (lambda x,t: ta.RSI(x,t)/100.),
                '#ema'                  : (lambda x,t: ta.EMA(x,t)),
                '#std'                  : (lambda x,t: ta.STDDEV(x,t)),
                '#min'                  : (lambda x,t: ta.MIN(x,t)),                  
                '#max'                  : (lambda x,t: ta.MAX(x,t))
                }

  
# =============================================================================
# 
# =============================================================================
    def _getTechnicals(self,theGroup):
        o,h,l,c,v = _getOHLCV(g)
        
        #myRSI(g,'close',[20,50,100])
        pass

# =============================================================================
# 
# =============================================================================
    def _mainLoop(self):
        df=self.itsDataframe
        logger=self.itsLogger
        
        theFrames=list()
        g=df.groupby(by=[self.itsTickerName])
        for name, group in g:
            logger.info('processing :<%s>: %s items',name,len(group))
            theFrames.append(self._getTechnicals(group))
        
# =============================================================================
# 
# =============================================================================
def test():
      x = FT_Xform()
 
if __name__ == "__main__":
    test()            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        