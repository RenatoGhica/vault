import pandas as pd
import numpy as np
from datetime import datetime
import random

from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
import keras.utils
from sklearn import preprocessing


import logging
import sys
            
class FT_xform:


    def __init__(self,theCalendar=holidays.US()):
        #self.itsReturnBuckets=np.linspace(-1,1,3)
        self.itsReturnBuckets=np.linspace(-1,1,3)
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
        