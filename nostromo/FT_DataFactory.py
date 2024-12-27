import pandas as pd
import numpy as np
from datetime import datetime
import random
import itertools

from six.moves.urllib.request import urlopen
from six.moves.urllib.parse import urlencode
import keras.utils
from sklearn import preprocessing

import dask.dataframe as dd

import logging
import sys
            
class FT_DataFactory:

    def slidingWindow2(self,sequence,winSize,step=1):
        numOfChunks = ((len(sequence)-winSize)/step)+1
        for i in range(0,numOfChunks*step,step):
            yield sequence[i:i+winSize]

    def slidingWindow(self,sequence,winSize,step=1):
        #print '[',len(sequence),winSize,step,']'
        numOfChunks=0
        try: it = iter(sequence)
        except TypeError:
            raise Exception("**ERROR** sequence must be iterable.")
        if not ((type(winSize) == type(0)) and (type(step) == type(0))):
            raise Exception("**ERROR** type(winSize) and type(step) must be int.")
        if step > winSize:
            raise Exception("**ERROR** step must not be larger than winSize.")
        if winSize > len(sequence):
            #raise Exception("**ERROR** winSize must not be larger than sequence len.")
            numOfChunks=0
        else:
            numOfChunks = ((len(sequence)-winSize)/step)+1
        #print 'chunks:',numOfChunks

        for i in range(0,numOfChunks*step,step):
            yield sequence[i:i+winSize]
# =============================================================================
# 
# =============================================================================
    def __init__(self,theFile='../data/stocks/stocks.csv', theTimeSeriesColumn='date',theTickerColumn='ticker'):
        self.timeSeriesColumn=theTimeSeriesColumn
        self.theTickerColumn=theTickerColumn        
        self.itsFileSource=theFile

        self.df=pd.DataFrame()
        self.theDataFilters=dict()
        
        self.colSeparator='|' # same as the one in FT_Model
        self.itsColumns=[]
        
        logging.basicConfig(level=logging.INFO,format='[%(asctime)s][%(levelname)-8s] -> %(message)s',)
        self.logger = logging.getLogger('DataFactory')
# =============================================================================
# 
# =============================================================================\    
    def info(self):
        logger = self.logger
        df=self.df
        
        
        print df.info()
        logger.info('grouping data')
        g = df.groupby(self.theTickerColumn)
        n_instruments = len(g)
        n_rows=len(df)
        logger.info('<%s rows, %s instruments loaded>',n_rows,n_instruments)
        if n_instruments < 100:
            logger.error('\tonly %s stocks loaded',n_instruments)   
            
        for colName in df.columns:
            c=df[colName]
            minV=c.min()
            maxV=c.max()
            logger.info('loaded {%s}\t[%s .. %s]',colName,minV,maxV)
            
# =============================================================================
#             
# =============================================================================
    def cleanup(self):
        df=self.df
        g = df.groupby(by=[self.theTickerColumn])
        for name, group in g:
            minH=group['high'].min()            
            
            if minH < 0:
                df=df[df.ticker==name]

# =============================================================================
# 
# =============================================================================
    def initTimeSeries(self):
        self.df[self.timeSeriesColumn] = pd.to_datetime(self.df[self.timeSeriesColumn])        
        self.df.set_index([self.theTickerColumn], inplace=False)    
        self.df.set_index([self.timeSeriesColumn], inplace=False)    
        
    def applyFilters(self):
        #df=self.df
        for key, val in self.theDataFilters.iteritems():
            #print '[filter] applying {} filter with value {}'.format(key,val)
            self.df=self.df[self.df[key].isin(val)]
            
    def getRandomTickers(self,theTickers,num=100):        
        results = random.sample(theTickers,len(theTickers))
        l=min(num,len(theTickers))
        return results[0:l]
        
    def getUniqueTickers(self):
        self.loadData()
        results = self.df.ticker.unique()
        return results
# =============================================================================
#         
# =============================================================================
    #def loadData(self,maxRows=10000000,theHeader=0):
    def loadData(self,theHeader=0):
        logger=self.logger
        theTypes={'open'    : np.float64,'high'    : np.float64, 'low'    : np.float64, 'close'    : np.float64, 'volume'    : np.float64, 
                  'adj_open': np.float32,'adj_high': np.float32, 'adj_low': np.float32, 'adj_close': np.float32, 'adj_volume': np.float32 }

        logger.info('loading data..')
        #self.df=pd.read_csv(self.itsFileSource,nrows=maxRows,parse_dates=True, header=theHeader,dtype=theTypes)
        #self.df=pd.read_csv(self.itsFileSource,nrows=maxRows,parse_dates=True, header=theHeader,dtype=theTypes)
        self.df=pd.read_csv(self.itsFileSource,parse_dates=True, header=theHeader,dtype=theTypes,memory_map=True,engine='c')
        #mydd =dd.read_csv(self.itsFileSource,parse_dates=True, header=theHeader,dtype=theTypes)
        logger.info('done.')
        #logger.info('converting to df')
        #self.df=mydd.compute()
        #logger.info('done.')
        #del mydd
        
        #for yahoo only
        if 'adj_open' in self.df.columns:
            logger.info('...found yahoo data')
            self.df.drop(['open','high','low','close','volume'],axis=1,inplace=True)
            self.cols_yahoo={'adj_open':'open','adj_high':'high','adj_low':'low','adj_close':'close','adj_volume':'volume'}
            self.df.rename(columns=self.cols_yahoo,inplace=True)
        
        self.initTimeSeries()
        self.applyFilters()
        self.info()
        self.cleanup()
        logger.info('done!')
        return self.df
        #print self.df

# # =============================================================================
# =============================================================================
#         print '[loaddata] removing rows <= 0...'
#         df=self.df
#         print df.shape
#         df.drop(df[df.high <=0].index,inplace=True)
#         print df.shape
#         df.drop(df[df.low <=0].index,inplace=True)
#         print df.shape
#         df.drop(df[df.close <=0].index,inplace=True)
#         print df.shape
#         print '[done]'
# 
# =============================================================================
    def getData(self):
        return self.df
        
    def setTickerFilter(self,theTickers):
        self.theDataFilters[self.theTickerColumn]=np.sort(np.unique(theTickers))
        return self.theDataFilters
    
    
        #pd.date_range(start='20170201',end='20170401
    def setDateFilter(self,theStartDate, theEndDate):
        self.theDataFilters[self.timeSeriesColumn]=pd.date_range(start=theStartDate,end=theEndDate)
        print '[setDateFilter] <{} to {}>'.format(min(self.theDataFilters[self.timeSeriesColumn]),max(self.theDataFilters[self.timeSeriesColumn]))
        
#        for key, val in self.theDataFilters.iteritems():
#            print '****',key,val
        return self.theDataFilters

    def setDateFilter_old(self,theDates):
        self.theDataFilters[self.timeSeriesColumn]=theDates
        print '[setDateFilter]'
        for key, val in self.theDataFilters.iteritems():
            print '****',key,val
        return self.theDataFilters

# =============================================================================
#             
# =============================================================================

    def normalize(self, df):
        #print df.head
        rsi_cols=[col for col in df.columns if '#rsi' in col]
        #print rsi_cols
        scaler = preprocessing.MinMaxScaler()
        df[rsi_cols]=scaler.fit_transform(df[rsi_cols])
        #df.head()
        return df
        
        
        
#    def updateColumns(self):
#        self.itsColumns = self.df.columns
        
# =============================================================================
    def scale_group(self,rawpoints, high=1.0, low=-1.0):
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng)
# =============================================================================

    def getColumnsIndex(self,theSearchPattern=None,skip=True):
        itsCols=self.df.columns
        #print itsCols
        
        if (theSearchPattern==None):
            result=[itsCols.get_loc(col) for col in itsCols]
        else:
            result=[itsCols.get_loc(col) for col in itsCols if theSearchPattern in col]
        
        if (skip):
           # print 'gettins skipped'
            s=self.getSkippedCols()
            result=[(x - s) for x in result]
            #print result
        #print result            
        return result

    def getColumns(self,theSearchPattern=None):
        itsCols=self.df.columns
        if (theSearchPattern==None):
            result=itsCols
        else:
            result=[col for col in itsCols if theSearchPattern in col]
        return result
    

    def getSkippedCols(self):
        df=self.df
        SKIP_COLS = df.columns.get_loc(self.colSeparator) + 1
        return SKIP_COLS
        
        
    def getModelData(self, theWindowSize=30,theTimeStep=1, thePredictions=5):

        df=self.df
        SKIP_COLS = df.columns.get_loc(self.colSeparator) + 1

        self.itsColumns = self.df.columns[SKIP_COLS:-1]

        df.sort_values(by=[self.theTickerColumn,self.timeSeriesColumn],ascending=[True,True],inplace=True)
        g=df.groupby(by=[self.theTickerColumn])
        
        #cols = df.columns[SKIP_COLS:-1]
        
        
        model_data=list()
        #grouped_data=dict()
        self.logger.info('<loading tickers..>')
        for name, group in g:
            theDataLen=0
            #print 'processing: [{} with length:{}]'.format(name,len(group))
            gen=self.slidingWindow(group.iloc[:,SKIP_COLS:],theWindowSize,theTimeStep)
            local_data=list()
            for theData in gen:                
                if (len(theData) >= theWindowSize):
                    #d=self.normalize(theData)
                    #print 'APPENDING:',len(theData)
                    local_data.append(theData.values)
                    theDataLen+=len(local_data)
                    #print 'local_data len:',len(local_data)
                else:
                    #print len(theData)
                    continue
            l=len(local_data)
            n=(l % theWindowSize)
            #print 'processing: [{} <{},{},{}>]'.format(name,len(group),l,(l-n))
            del local_data[:n]
            #grouped_data[name]=local_data
            
            #new
            model_data.append(local_data)
            #model_data=model_data + local_data
            
        self.logger.info('processed: [%s]',theDataLen)
        #new
        results=list(itertools.chain.from_iterable(model_data))
        
        results=np.asarray(results)
        
        del model_data
        self.itsColumns = self.df.columns[SKIP_COLS:-1]
        #print df.info(memory_usage='deep')
        #print df.info()
        return self.itsColumns,results
# =============================================================================
# =============================================================================
# 
# =============================================================================
    def getXY(self,theData, theSplit=0.2, thePredictions=1):    
        #TODO : optimize
        print theData.shape
        g=np.asarray(theData)
        print g.shape
        theTimesteps=g.shape[1]
        all_X, all_Y = g[:,:, :-thePredictions], g[:,-1, -thePredictions]           
        
        print all_X.shape, all_Y.shape
        
        l=len(all_X)
        print l
        l-=l % theTimesteps
        print l
        L=  int(l/theTimesteps)
        print L
        L=int(L*0.2)*theTimesteps
        print L
        if L==0: L=theTimesteps
    
        train_size=l-L
        print l,L, train_size
        
        X=all_X[0:train_size]
        Y=all_Y[0:train_size]
        
        test_X=all_X[train_size:]
        test_Y=all_Y[train_size:]
    
        print g[0][0]
        print X[0][0]
        print Y[0]
        print test_X[0][0]
        print test_Y[0]
    
        print '*******************'
        print X[0:10]
        print Y[0:10]
        #print X.shape, Y.shape
        #print test_X.shape, test_Y.shape
    
        return X, Y, test_X, test_Y
# =============================================================================
# 
# =============================================================================
    def splitData(self,theGroupedData, theWindowSize=30, thePredictions=1, theSplit=0.1):
    
        model_data=[]        
        for key, value in theGroupedData.iteritems():
            grouped_data=np.asarray(value)
            split=int(len(grouped_data)/theWindowSize)
            split=int(split*theSplit)
            split=split*theWindowSize

            
            split=(len(grouped_data)-split)
            
            train_X, train_Y = grouped_data[0:split,:, :-thePredictions], grouped_data[0:split,-1, -thePredictions:]       
            test_X, test_Y   = grouped_data[split:,:, :-thePredictions], grouped_data[split:,-1, -thePredictions:]               
            #print len(grouped_data)
            #print train_X.shape, train_Y.shape
            #print test_X.shape, test_Y.shape
            model_data.append([train_X,train_Y,test_X,test_Y])
    
        return model_data        

# =============================================================================
# 
# =============================================================================
    def _genTickers(self,theFile):
        d1=pd.read_csv(theFile)
        tickers=sorted(d1[self.theTickerColumn].unique())
        d1=pd.DataFrame(tickers,columns=[self.theTickerColumn])
        d1.to_csv('../data/tickers.csv')
        return tickers
    
    def _loadTickers(self,theFile='../data/tickers.csv'):
        d1=pd.read_csv(theFile)
        return d1[self.theTickerColumn]
    
    def genYear(self,theYearStart, theYearEnd, theTickers):
        data = FT_DataFactory()
        sd = datetime(theYearStart, 1, 1).date()
        ed = datetime(theYearEnd, 12, 31).date()
        
        cols=[self.theTickerColumn,self.timeSeriesColumn, 'open', 'high', 'low', 'close', 'volume']
        df_master=pd.DataFrame(columns=cols)
        
        for ticker in theTickers:
            print 'processing:',ticker
            q = data.quotes_historical_google(ticker,sd,ed)
            if len(q)==0:
                continue
            df=pd.DataFrame(q)
            #df.sort(['date'],ascending=[1])
            df[self.theTickerColumn]=ticker
            df_master=pd.concat([df_master,df])
        
        df_master=df_master[cols]
        df_master=df_master.sort_values([self.theTickerColumn,self.timeSeriesColumn],ascending=[1,1])
        
        
        f_name='../data/'+str(theYearStart)+'-'+str(theYearEnd)+'_stocks.csv'
        print f_name
        df_master.to_csv(f_name,float_format='%.3f')
    
    def createData(self):
        tickers = self._loadTickers()
        #tickers = self._getTickers()
        self.genYear(2007,2017,tickers)
            
        
    def quotes_historical_google(self,symbol, start_date, end_date):
        """Get the historical data from Google finance.
    
        Parameters
        ----------
        symbol : str
            Ticker symbol to query for, for example ``"DELL"``.
        start_date : datetime.datetime
            Start date.
        end_date : datetime.datetime
            End date.
    
        Returns
        -------
        X : array
            The columns are ``date`` -- date, ``open``, ``high``,
            ``low``, ``close`` and ``volume`` of type float.
        """
        params = {
            'q': symbol,
            'startdate': start_date.strftime('%Y-%m-%d'),
            'enddate': end_date.strftime('%Y-%m-%d'),
            'output': 'csv',
        }
        try:
    
            url = 'https://finance.google.com/finance/historical?' + urlencode(params)
            response = urlopen(url)
            dtype = {
                'names': ['date', 'open', 'high', 'low', 'close', 'volume'],
                'formats': ['object', 'f4', 'f4', 'f4', 'f4', 'f4']
            }
            converters = {
                0: lambda s: datetime.strptime(s.decode(), '%d-%b-%y').date()}
        
    
            data = np.genfromtxt(response, delimiter=',', skip_header=1,
                                 dtype=dtype, converters=converters,
                                 missing_values='-', filling_values=-1)
        except:
            return []
        
        min_date = min(data['date'])
        max_date = max(data['date'])
        
        #min_date = min(data['date'])
        #max_date = max(data['date'])
    
        start_end_diff = (end_date - start_date).days
        min_max_diff = (max_date - min_date).days
        data_is_fine = (
            start_date <= min_date <= end_date and
            start_date <= max_date <= end_date and
            start_end_diff - 7 <= min_max_diff <= start_end_diff)
    
        #if not data_is_fine:
        if False:
            message = (
                'Data looks wrong for symbol {}, url {}\n'
                '  - start_date: {}, end_date: {}\n'
                '  - min_date:   {}, max_date: {}\n'
                '  - start_end_diff: {}, min_max_diff: {}'.format(
                    symbol, url,
                    start_date, end_date,
                    min_date, max_date,
                    start_end_diff, min_max_diff))
            raise RuntimeError(message)
        return data


# =============================================================================
# 
# =============================================================================
def test():
# =============================================================================
      data = FT_DataFactory()
#      data.createData()
# # =============================================================================
      data.setDateFilter('2010-01-01','2017-01-01')
      data.loadData()     
      data.info()
      data.getData().head()
 
if __name__ == "__main__":
    test()           