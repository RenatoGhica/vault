import pandas as pd
import numpy as np
import timeit
import time
import ast
import sys
from timeit import Timer

def _setDisplay():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
def myfunc(x):
        return pd.Series(x,index=[x.keys()])
    
def rebalanceAccounts(theAccounts, theModels, theSecurities):
    
    theGrid=theAccounts.merge(theModels,how='inner')

    if (True):
        cols=np.unique(np.concatenate(theGrid.allocations.apply(lambda x: x.keys()).values))
        theTargets=pd.DataFrame()
        #theTargets=pd.concat([theTargets,theGrid.ix[:,0]])
        theTargets=theGrid['accountNum']
        for col in (cols):
            theTargets=pd.concat([theTargets,pd.Series(theGrid.allocations.apply(lambda x: dict(x).get(col,0)),name=col)],axis=1)
        
        #theTargets=pd.concat([theGrid.ix[:,0],theGrid.ix[:,5:]],axis=1)

        
        #print theTargets.head()
        
        cols=np.unique(np.concatenate(theGrid.positions.apply(lambda x: x.keys()).values))
        print cols
        
        theHoldings=pd.DataFrame()
        theHoldings=theGrid['accountNum']
        for col in (cols):
            theHoldings=pd.concat([theHoldings,pd.Series(theGrid.positions.apply(lambda x: dict(x).get(col,0)),name=col)],axis=1)
        #theHoldings=pd.concat([theGrid.ix[:,0],theGrid.ix[:,5:]],axis=1)        
        


        theSecurities=theSecurities.sort_index()
        theSecurities=theSecurities.T
        
    
        theHPrices=theSecurities[list(theHoldings.ix[:,1:].columns)]
        theHoldings[list(theHPrices.columns.values)]=theHoldings[list(theHPrices.columns.values)].mul(theHPrices.ix[0],axis=1)
        theHoldings['balance']=theHoldings.sum(axis=1)

        theTargets.ix[:,1:-1]=theTargets.ix[:,1:-1].mul(theHoldings['balance']/100.,axis=0)
        results=pd.concat([theHoldings.ix[:,0],theHoldings.ix[:,-1],theHoldings.ix[:,1:-1].rsub(theTargets.ix[:,1:-1],fill_value=0)],axis=1)


        print theGrid,'\n',theHoldings.head(),'\n',theTargets.head(), '\n', results.head()

    #theTargets=pd.concat([theGrid,theGrid.allocations.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1).fillna(0)
    if (False):
    
        theTargets=pd.concat([theGrid,theGrid.allocations.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1).fillna(0)
        theTargets=pd.concat([theTargets.ix[:,0],theTargets.ix[:,5:]],axis=1)
        
        theHoldings=pd.concat([theGrid,theGrid.positions.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1).fillna(0)
        theHoldings=pd.concat([theHoldings.ix[:,0],theHoldings.ix[:,5:]],axis=1)
    
        
        theSecurities=theSecurities.sort_index()
        theSecurities=theSecurities.T
        
    
        theHPrices=theSecurities[list(theHoldings.ix[:,1:].columns)]
        theHoldings[list(theHPrices.columns.values)]=theHoldings[list(theHPrices.columns.values)].mul(theHPrices.ix[0],axis=1)
        theHoldings['balance']=theHoldings.sum(axis=1)
        
        
        theTargets.ix[:,1:-1]=theTargets.ix[:,1:-1].mul(theHoldings['balance']/100.,axis=0)
        results=pd.concat([theHoldings.ix[:,0],theHoldings.ix[:,-1],theHoldings.ix[:,1:-1].rsub(theTargets.ix[:,1:-1],fill_value=0)],axis=1)
        #results=theHoldings.ix[:,1:-1].rsub(theTargets.ix[:,1:-1],fill_value=0).combine_first(theTargets.ix[:,1:])


        

def main():
    
    _setDisplay()
    
    n=(100 if (len(sys.argv))==1 else int(sys.argv[1]))
    print n
        
    sec = pd.read_csv('data/rebal_sec.csv',sep=',')
    acc = pd.read_csv('data/rebal_acc.csv',sep=',',nrows=n,converters={'positions':ast.literal_eval})
    mod = pd.read_csv('data/rebal_mod.csv',sep=',',nrows=n,converters={'allocations':ast.literal_eval})
    
    sec.set_index(['ticker'], inplace=True)
    mod.set_index(['model_id'],inplace=False)
    #acc.set_index(['accountNum'],inplace=False)
    

    
    t0=time.time()
    #t = Timer(lambda: rebalanceAccounts(acc[0:n], mod, sec))
    rebalanceAccounts(acc[0:n], mod, sec)
    t1=time.time()
    print t1-t0
    #print t.timeit(1),' seconds'
    
    #results.head()
    
if __name__ == '__main__':
    main()    
    

    

    
    