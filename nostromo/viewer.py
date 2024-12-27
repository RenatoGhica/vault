import numpy as np
import pandas as  pd

from math import sqrt
from FT_Model import FT_Model
from FT_DataFactory import FT_DataFactory

from keras.layers import BatchNormalization, Input, Dense, LSTM, Dropout, Activation, TimeDistributed, Flatten, RepeatVector, BatchNormalization, Conv1D, Conv2D
#from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D
#from keras.models import Sequential, Model
from keras import optimizers, metrics
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
import keras.utils
from keras.models import Sequential, Model

#from sklearn.metrics import mean_squared_error, mean_absolute_error
#from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras import backend as K

from FT_CNN_Model import FT_CNN_Model

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
#from sklearn.preprocessing import MinMaxScaler
import sklearn

global current_record

# =============================================================================
current_record=0
WINDOW_SIZE=128
TIME_STEP=5
PREDICTIONS=1
B_WINDOW_SIZE=64
N_TICKERS=500
# =============================================================================
def scale_linear_bycolumn(rawpoints, high=1.0, low=-1.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)
# =============================================================================
def showit(theFactory,theData,theGroups):
    plt.style.use('seaborn-whitegrid')
    fig, axes = plt.subplots(nrows=len(theGroups), ncols=1,sharex=False,squeeze=False)
    plt.clf()
    
    for idx,g in enumerate(theGroups):
        col_idx = theFactory.getColumnsIndex(g,True)
        #print col_idx
        col_names = theFactory.getColumns(g)
        #print col_names
        v=len(colGroups)
        plt.subplot(v,1,idx+1)
    
    
        m = theData[:,col_idx]
        orig_shape=theData[:,col_idx].shape            
        
        print m.shape,orig_shape
        
        m2=scale_linear_bycolumn(m.flatten())
        #m3=keras.utils.normalize(m2,axis=0)
        #m4=scale_linear_bycolumn(m3)
        #print m2.shape
        m5=m2.reshape(orig_shape)
        #print m3.shape

        v=m5
        for idx2, d in enumerate(col_idx):        
            s=col_names[idx2]
            v_name,v_col,v_num,v_type=s.split('.')
            l='.'.join([v_name,v_col,v_num])
            
            l_norm=l+'.norm'
            #print l_norm
            plt.plot(v[:,idx2],label=l)
            #plt.plot(m3)

            plt.title(v_name,loc='left')
            plt.legend(bbox_to_anchor=(1.0,1.0),fontsize='x-small')

#    ax1=plt.gca()
#    ax1.relim()
#    ax1.autoscale_view()
    plt.show()
    plt.clf()
    plt.close()

d2 = FT_DataFactory('../data/modeldata_stage3.csv')
d2.setTickerFilter(['AAL'])
d2.setDateFilter(theStartDate='20100101',theEndDate='20170101')
d2.loadData()

theDataColumns, model_data = d2.getModelData(WINDOW_SIZE, 1, 1)

#colGroups = ['rsi.high','ema.high','angle.high','mfi','value.high',col]
colGroups = ['rsi.high','ema.high','raw_value.high']
#print norm1
#v=np.fft.fft(norm1).real

#fig, ax = plt.subplots(nrows=len(colGroups), ncols=1,sharex=True)
showit(d2,model_data[0,:,:],colGroups)

#model_data2=keras.utils.normalize(model_data[0,:,:])

#showit(d2,model_data2[0,:,:],colGroups)

#theDataColumns2, model_data = d2.getModelData(WINDOW_SIZE, 1, 1)
#showit(d2,model_data[0,:,:],colGroups)




