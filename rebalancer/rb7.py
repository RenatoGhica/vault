import pandas as pd
import numpy as np
import timeit
import time
import ast
import sys
import datetime
from timeit import Timer

def _setDisplay():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
def myfunc(x):
        return pd.Series(x,index=[x.keys()])
    
def rebalanceAccounts(theAccounts, theModels, theSecurities):
    
    
    theModels=theModels.copy()
    theSecurities=theSecurities.copy()
    theAccounts=theAccounts.copy()
    theGrid=theAccounts.merge(theModels,how='inner') 
    
    theTargets=pd.DataFrame()
    theHoldings=pd.DataFrame()
    theTargets['accountNum']=theGrid['accountNum']
    theHoldings['accountNum']=theGrid['accountNum']
    
    print datetime.datetime.now().time()
    if (True):
        cols=np.unique(np.concatenate(theGrid.allocations.apply(lambda x: x.keys()).values))
        for col in (cols):
            theTargets=pd.concat([theTargets,pd.Series(theGrid.allocations.apply(lambda x: x.get(col,0)),name=col)],axis=1)
        print 1,datetime.datetime.now().time()    
        
        cols=np.unique(np.concatenate(theGrid.positions.apply(lambda x: x.keys()).values))
        print 1.5,datetime.datetime.now().time()   
        for col in (cols):
            #theHoldings=pd.concat([theHoldings,pd.Series(theGrid.positions.apply(lambda x: x.get(col,0)),name=col)],axis=1)
            
           
            #theHoldings.insert(1,col,pd.Series(theGrid.positions.apply(lambda x: x.get(col,0)),name=col))
            theHoldings[col]=pd.Series(theGrid.positions.apply(lambda x: x.get(col,0)),name=col)
            #pd.Series(theGrid.positions.apply(lambda x: dict(x).get(col,0)),name=col)
            #theGrid.positions.apply(lambda x: dict(x).get(col,0))
            #y=dict(x).get(col,0)
            #theGrid.positions.apply(lambda x: x)
        #exit(1)
        print 2,datetime.datetime.now().time()        
        #exit(1)
        theSecurities=theSecurities.sort_index().T
        #theSecurities=theSecurities.T
        print 3,datetime.datetime.now().time()
    
        theHPrices=theSecurities[list(theHoldings.ix[:,1:].columns)]
        theHoldings[list(theHPrices.columns.values)]=theHoldings[list(theHPrices.columns.values)].mul(theHPrices.ix[0],axis=1)
        theHoldings['balance']=theHoldings.sum(axis=1)
        print 4,datetime.datetime.now().time()
        
        theTargets.ix[:,1:]=theTargets.ix[:,1:].mul(theHoldings['balance']/100.,axis=0)
        print datetime.datetime.now().time()
        results=pd.concat([theHoldings.ix[:,0],theHoldings.ix[:,-1],theHoldings.ix[:,1:-1].rsub(theTargets.ix[:,1:-1],fill_value=0)],axis=1)
        print 5,datetime.datetime.now().time()
        #print theGrid,'\n',theHoldings.head(),'\n',theTargets.head(), '\n', results.head()
        print theGrid.ix[0:10,0:20]
        print 6,theHoldings.ix[0:10,0:20],'\n',theTargets.ix[0:10,0:20], '\n', results.ix[0:10,0:20]
    return results

def main():
    
    _setDisplay()
    
    n=(100 if (len(sys.argv))==1 else int(sys.argv[1]))
    print n
        
    sec = pd.read_csv('data/rebal_sec.csv',sep=',')
    acc = pd.read_csv('data/rebal_acc.csv',sep=',',nrows=n,converters={'positions':ast.literal_eval})
    mod = pd.read_csv('data/rebal_mod.csv',sep=',',nrows=n,converters={'allocations':ast.literal_eval})
    
    sec.set_index(['ticker'], inplace=True)
    mod.set_index(['model_id'],inplace=False)
    acc.set_index(['accountNum'],inplace=False)
    

    
    t0=time.time()
    #t = Timer(lambda: rebalanceAccounts(acc[0:n], mod, sec))
    results=rebalanceAccounts(acc[0:n], mod, sec)
    
    t1=time.time()
    print t1-t0
    print results.ix[0:10,0:20]
    #print t.timeit(1),' seconds'
    
    #accounts with apple
    #ticker='AAPL'
    #if '_OTC_999' in results: (results[results['_OTC_999'] > 0])
    #print results.columns
    #print '***\n', results.loc[np.abs(results['_OTC_984']) > 0]
    #orders = results.apply(results[x],results[x].value)
    
    #print x
    #results.head()
    
if __name__ == '__main__':
    main()    
    

    

    
    