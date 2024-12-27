import string
import random
import pandas as pd
import numpy as np

def _setDisplay():
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(xrange(1, total), n - 1))
    results= [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    np.reshape(results, (len(results),1))
    
    return results

def getAccountNumber(theCount=10,theLength=8):  
    return pd.Series(''.join(random.choice(string.digits) for _ in range(theLength)) for _ in range(theCount))

def getRandomModelAllocations(secs,minItems=2, maxItems=3):

    maxItems=len(secs) if maxItems > len(secs) else maxItems
    model_stocks=random.sample(secs,np.random.randint(minItems,maxItems))
    model_alloc=constrained_sum_sample_pos(len(model_stocks),100)
    #esults =
    
    results = dict(zip(model_stocks, model_alloc))
    return results
    
def getRandomPositions(secs,minItems=2, maxItems=3):

    maxItems=len(secs) if maxItems > len(secs) else maxItems
    model_stocks=random.sample(secs,np.random.randint(minItems,maxItems))
    #model_alloc=constrained_sum_sample_pos(len(model_stocks),100)
    model_alloc=(np.random.randint(1,10,len(model_stocks))*100)

    results = dict(zip(model_stocks, model_alloc))
    #print results
    return results

def rebalanceAccounts(theAccounts, theModels, theSecurities):
    
    #print theModels.loc[theAccounts['model_id']]
    
    theGrid=pd.DataFrame({'accountNum':theAccounts['accountNum'],
                          'balance' : theAccounts['balance'],
                          'model' :   theAccounts['model_id']
                          
                          })
    theGrid.set_index(['accountNum'], inplace=True)
    
    theGrid['allocations']=theModels.loc[theAccounts['model_id']].values
    theGrid['positions']=theAccounts['positions'].values
    
    secs = theSecurities.T.columns.values
    
    
    target=theGrid.copy()
    actuals=theGrid.copy()
    #target=pd.concat([target,pd.DataFrame(columns=[secs])]).fillna(0)
    #ctuals=pd.concat([actuals,pd.DataFrame(columns=[secs])]).fillna(0)
    #print target,actuals

    #print target
    target=pd.concat([target,target.allocations.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1).fillna(0)
    actuals=pd.concat([actuals,actuals.positions.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1).fillna(0)

    target=target.drop(['model','allocations','positions'],axis=1)
    actuals=actuals.drop(['model','allocations','positions'],axis=1)
    
    #actuals=pd.concat([actuals,pd.DataFrame(columns=[secs])]).fillna(0)

    theSecurities=theSecurities.sort_index()
    theSecurities=theSecurities.T

    #print theSecurities
    #print target#, target
    actuals=actuals.mul(theSecurities.ix[0]).fillna(0)    
    actuals.balance=actuals.sum(axis=1)
    #print actual_bals       
    target.balance=actuals.balance
    #target.ix[:,1:]=target.mul(target.balance/100,axis=0)

    target=target.reindex(actuals.index)
    
    target.ix[:,1:]=target.multiply(target.balance/100, axis='index')
    #print  target
    results = (target - actuals).fillna(0)
    results.balance=results.sum(axis=1)
    
    print results
    results = results - actuals
    results.balance=results.sum(axis=1)
    
    return results
    
    
def main():
    _setDisplay()
    
   
# =============================================================================
#     sec=pd.DataFrame({'ticker': np.reshape(theSecuritiesUniverse,len(theSecuritiesUniverse),1), 
#                       'price': np.random.choice(range(10,50),len(theSecuritiesUniverse))})
#     
# =============================================================================
    sec=pd.DataFrame(zip(theSecuritiesUniverse, np.random.choice(range(10,50),len(theSecuritiesUniverse)))
                    ,columns=['ticker','price'])

    #mod=pd.DataFrame({'model_id': pd.Series('model_'+str(i) for i in range(NUM_ACCOUNTS)), 
    #                  'allocations': pd.Series(getRandomModelAllocations(theSecuritiesUniverse,3,6) for _ in range(NUM_ACCOUNTS))})
    mod=pd.DataFrame({'model_id'    :pd.Series(map('model_id_{}'.format,xrange(NUM_ACCOUNTS))),
                      'allocations' :pd.Series(getRandomModelAllocations(theModelSecurities,3,6) for _ in range(NUM_ACCOUNTS))
                      })
    
    
    acc=pd.DataFrame({
            
                      'accountNum':pd.Series(getAccountNumber(NUM_ACCOUNTS)),
                      'balance':   0,
                      'positions': pd.Series(getRandomPositions(theRandomSecurities,2,4) for _ in range(NUM_ACCOUNTS)),
                      'model_id':  mod['model_id']
                      })



    sec.set_index(['ticker'], inplace=True)
    mod.set_index(['model_id'],inplace=True)
    acc.set_index(['accountNum'],inplace=False)
      
    #print getRandomPositions()
    #print acc
    
    results = rebalanceAccounts(acc, mod, sec)
    results.head()
    
if __name__ == '__main__':
    random.seed(7)
    np.random.seed(7)
    theModelSecurities = ['IBM','AAPL','GOOG','MSFT','C','X','T','BAC']
    theRandomSecurities = map('_OTC_{}'.format,xrange(5))
    theSecuritiesUniverse=theModelSecurities+theRandomSecurities
    NUM_ACCOUNTS=100000
    INITIAL_BAL=100000
    main()    
    

    

    
    