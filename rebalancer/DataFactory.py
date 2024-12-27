import string
import random
import pandas as pd
import numpy as np
import sys

def constrained_sum_sample_pos(n, total):
    """Return a randomly chosen list of n positive integers summing to total.
    Each such list is equally likely to occur."""

    dividers = sorted(random.sample(xrange(1, total), n - 1))
    results= [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    np.reshape(results, (len(results),1))
    
    return results

def getRandomString(theSet, theCount=10,theLength=8):  
    return pd.Series(('A'+''.join(random.choice(theSet) for _ in range(theLength))) for _ in range(theCount))

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

def main():
    sec=pd.DataFrame(zip(theSecuritiesUniverse, np.random.choice(range(10,50),len(theSecuritiesUniverse))),columns=['ticker','price'])

    mod=pd.DataFrame({'model_id'    :pd.Series(map('model_id_{}'.format,xrange(NUM_ACCOUNTS))),
                      'allocations' :pd.Series(getRandomModelAllocations(theModelSecurities,3,6) for _ in range(NUM_ACCOUNTS))
                      })
    
    
    
    acc=pd.DataFrame({
            
                      'accountNum':pd.Series(getRandomString(string.digits,NUM_ACCOUNTS)),
                      'balance':   0,
                      'positions': pd.Series(getRandomPositions(theRandomSecurities,2,4) for _ in range(NUM_ACCOUNTS)),
                      'model_id':  mod['model_id']
                      })
    
    
    #sec.set_index(['ticker'], inplace=True)
    #mod.set_index(['model_id'],inplace=True)
    #acc.set_index(['accountNum'],inplace=False)
    
    sec.to_csv('data/rebal_sec.csv',sep=',',index=False)
    acc.to_csv('data/rebal_acc.csv',sep=',',index=False)
    mod.to_csv('data/rebal_mod.csv',sep=',',index=False,columns=['model_id','allocations'])
    
    

if __name__ == '__main__':
    random.seed(7)
    np.random.seed(7)
    theModelSecurities = ['IBM','AAPL','GOOG','MSFT','C','X','T','BAC']
    
    theRandomSecurities = map('_OTC_{}'.format,xrange(1000))
    theRandomSecurities.append('GOOG') # assume some people have securities from the models already
    theRandomSecurities.append('AAPL')
    theSecuritiesUniverse=theModelSecurities+theRandomSecurities
    NUM_ACCOUNTS=(100 if (len(sys.argv)==0) else int(sys.argv[1]))
    INITIAL_BAL=100000
    main()    
    