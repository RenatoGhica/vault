import numpy as np
import pandas as  pd
import datetime
import time
import sys, signal
import random

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

#from keras.utils.np_utils import to_categorical

from FT_CNN_Model import FT_CNN_Model
 
stocks = ['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS',  
               'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE',  
               'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON',  
               'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY',  
               'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX',  
               'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB',  
               'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK',  
               'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE',  
               'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS',  
               'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR',  
               #'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA',  
               'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE',  
               'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS',  
               'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GRMN', 'GD', 'GE', 'GGP', 'GIS',  
               'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG',  
               'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN',  
               'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ',  
               'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM', 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB',  
               'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB',  
               'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN',  
               'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS',  
               'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP',  'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI',  
               'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR',  
               'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD',  
               'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM',  
               'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP',  
               'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM',  
               'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY',  
               'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS',  
               'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR',  
               'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB',  
               'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

tickers_all1 = ['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS',  
               'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE',  
               'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON',  
               'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY',  
               'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX',  
               'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB',  
               'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK',  
               'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE']
 
o_list=['adadelta', 'SGD','rmsprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
tickersAll=['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG']
tickers=['BAC','C','GS','JPM','BK','AXP','WTW','AMZN','NFLX','ANF','GPS','AAPL','INTC','ABT', 'ABBV', 'ACN', 'ACE']



    
# =============================================================================
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        model.save()
        sys.exit(0)

# =============================================================================
def normalize(array, imin = -1, imax = 1):
    """I = Imin + (Imax-Imin)*(D-Dmin)/(Dmax-Dmin)"""

    dmin = array.min()
    dmax = array.max()


    array -= dmin;
    array *= (imax - imin)
    array /= (dmax-dmin)
    array += imin

    #print array[0]
# =============================================================================

WINDOW_SIZE=64
TIME_STEP=5
PREDICTIONS=1
B_WINDOW_SIZE=64
N_TICKERS=100

d = FT_DataFactory('../data/modeldata_stage3.csv')
d.setDateFilter(theStartDate='20100101',theEndDate='20170101')

print 'getting unique tickers...'
#all_symbols=d.getUniqueTickers()
all_symbols=stocks


print 'getting ',N_TICKERS,'random tickers...'
load_symbols=d.getRandomTickers(all_symbols,N_TICKERS)
T=int(len(load_symbols)*0.8)
tickersTrain,tickersTest = load_symbols[0:T],load_symbols[T:]
print 'actually got ',len(load_symbols),'random tickers...'
#tickersTest=load_symbols[T:]


print 'loading data..'
d.setTickerFilter(tickersTrain)

d.loadData()
#print d.df.memory_usage(True,True).sum()
#print sys.getsizeof(d.df)
#exit(1)
print 'done loading data!'
print d.df.shape


theDataColumns, model_data= d.getModelData(WINDOW_SIZE, TIME_STEP, PREDICTIONS)
print d.df['TARGET0'].value_counts()



samples=model_data.shape[0]
timesteps=model_data.shape[1]
features=model_data.shape[2]-PREDICTIONS

colGroups = ['raw_value']

orig_shape=model_data.shape
print model_data[0]
for g in colGroups:
    group = d.getColumnsIndex(g)
    print 'before:',model_data[0,:,group[0]]
    for i in range(model_data.shape[0]):
        v = model_data[i,:,group].flatten()
        normalize(v)
        #print v.shape
        model_data[i,:,group]=np.reshape(v,(1,model_data.shape[1],len(group)))
    print 'after:',model_data[0,:,group[0]]
exit(1)
model = FT_CNN_Model("mymodel",colGroups,timesteps,features,B_WINDOW_SIZE,d)

#top_N_sparse_categorical_accuracy = lambda y_true,y_pred: metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=2)


opt=optimizers.adam(lr=0.001)
#opt=optimizers.adam(lr=0.001)

#opt=optimizers.adadelta()
#optimizers.adagrad(lr=0.01)
#opt=optimizers.nadam(lr=0.002,schedule_decay=0.01)

signal.signal(signal.SIGINT, signal_handler)

model.compileModel(model_data, theLossFn='sparse_categorical_crossentropy',theOptimizerFn=opt,theMetrics=[metrics.sparse_categorical_accuracy])

scores = model.runModel(model_data,thePredictions=1, theSplit=0.2,theWindowSize=WINDOW_SIZE,theBatchSize=B_WINDOW_SIZE,theEpochs=100,isNormalizationEnabled=False)


m = model.getModel()
model.save()
#exit(1)
# =============================================================================
# 
# =============================================================================
d2 = FT_DataFactory('../data/modeldata_stage3.csv')
d2.setTickerFilter(tickersTest)
d2.setDateFilter(theStartDate='20100101',theEndDate='20170101')
d2.loadData()

print len(d2.df)
theDataColumns2, model_data2 = d2.getModelData(WINDOW_SIZE, TIME_STEP, 1)
print d2.df['TARGET0'].value_counts()
print model_data2.shape


#print d2.df.head()

samples=model_data2.shape[0]
timesteps=model_data2.shape[1]
features=model_data2.shape[2]-PREDICTIONS

opt=optimizers.adam(lr=0.001)

#model_test = FT_CNN_Model("mymodel",timesteps,features,B_WINDOW_SIZE,d2)

#model_test.compileModel(model_data2, theLossFn='sparse_categorical_crossentropy',theOptimizerFn=opt,theMetrics=[metrics.sparse_categorical_accuracy])
#model_test.load_weights()
X2, Y2, test_X2, test_Y2 = model.getXY(model_data2,0,1,B_WINDOW_SIZE,isCategorical=False)
print X2.shape, Y2.shape

scores_test=model.evaluate(X2,Y2,theBatchSize=WINDOW_SIZE)
print("\n\r %s: %.2f%%" % (model.get().metrics_names[1], scores_test[1]*100))

#print predictions

#import matplotlib.pyplot as plt
#plt.plot(Y2,label='actual')
#plt.plot(guess2,label='predicted')
#plt.show()

