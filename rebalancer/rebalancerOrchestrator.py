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
    #esults = map(lambda x,y: (x,y),model_stocks,model_alloc)
    results = dict(zip(model_stocks, model_alloc))
    return results
    
def getRandomPositions(minItems=2, maxItems=3):

    maxItems=len(theSecuritiesUniverse) if maxItems > len(theSecuritiesUniverse) else maxItems
    model_stocks=random.sample(theSecuritiesUniverse,np.random.randint(minItems,maxItems))
    #model_alloc=constrained_sum_sample_pos(len(model_stocks),100)
    model_alloc=np.random.randint(100,1000,len(model_stocks))
    results = map(lambda x,y: (x,y),model_stocks,model_alloc)
    results = dict(results)
    return results

def rebalanceAccounts(theAccounts, theModels, theSecurities):
    
    print theModels.loc[theAccounts['model_id']]
    
    theGrid=pd.DataFrame({'accountNum':theAccounts['accountNum'],
                          'balance' : theAccounts['balance'],
                          'model' :   theAccounts['model_id']
                          
                          })
    theGrid.set_index(['accountNum'], inplace=True)
    
    theGrid['allocations']=theModels.loc[theAccounts['model_id']].values
    
    
    s=theGrid['allocations']
    
    
    l = list(s)

    secList=np.unique(zip(*list(s)))
    #theGrid=theGrid.append(pd.DataFrame(columns=np.unique(((zip(*list(s)))))))
    theGrid2=pd.concat([theGrid,theGrid.allocations.apply(lambda x: pd.Series(x,index=[x.keys()]))],axis=1)
    theGrid2.fillna(0,inplace=True)
    print theGrid
    print theGrid2
    exit(1)
# =============================================================================
#     for col in range (len(secList)):
#          theGrid[secList[col]]=0
#          theGrid.set_index(secList[col])
#     
# =============================================================================
    g=theGrid.groupby('accountNum')['allocations']
    g.apply(lambda x:pd.DataFrame(x.iloc[0],columns=[secList]))

    for i in range (len(l)):
        #print l[i]
        for j in range (len(l[i])):
           l2=l[i][j][0]
           secList.add(l2)
    
    #print secList
# =============================================================================
#     for col in range (len(secList)):
#         theGrid[secList[col]]=0
#         theGrid.set_index(secList[col])
#          
#     #print theGrid
#     for i,row in theGrid.iterrows():
#         m = row[2]
#         for s in range (len(m)):
#             theGrid[m[s][0]][i]=m[s][1]
# =============================================================================
    print theGrid
    
    
    
    
def main():
    _setDisplay()
    
   
    sec=pd.DataFrame({'cusip': np.reshape(theSecuritiesUniverse,len(theSecuritiesUniverse),1), 
                      'price': np.random.choice(range(10,50),len(theSecuritiesUniverse))})
    
    mod=pd.DataFrame({'model_id': pd.Series('model_'+str(i) for i in range(NUM_ACCOUNTS)), 
                      'allocations': pd.Series(getRandomModelAllocations(theSecuritiesUniverse,3,6) for _ in range(NUM_ACCOUNTS))})
    
    acc=pd.DataFrame({
                      'accountNum':pd.Series(getAccountNumber(NUM_ACCOUNTS)),
                      'balance':   INITIAL_BAL,
                      'positions': pd.Series((map(lambda x,y: (x,y),['#CASH'],[INITIAL_BAL])) for _ in range(NUM_ACCOUNTS)),
                      'model_id':  mod['model_id']
                      })



    sec.set_index(['cusip'], inplace=False)
    mod.set_index(['model_id'],inplace=True)
    acc.set_index(['accountNum'],inplace=False)
      
    print getRandomPositions()
    #print acc
    
    rebalanceAccounts(acc[0:30], mod, sec)
    
if __name__ == '__main__':
    random.seed(7)
    np.random.seed(7)
    theSecuritiesUniverse = ['IBM','AAPL','GOOG','MSFT','C','X','T','BAC']
    NUM_ACCOUNTS=20
    INITIAL_BAL=100000
    main()    
    

    

    
    