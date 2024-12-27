import string
import random
import pandas as pd
import numpy as np
import sys

def constrained_sum_sample_pos(n, total):
    dividers = sorted(random.sample(xrange(1, total), n - 1))
    results= [a - b for a, b in zip(dividers + [total], [0] + dividers)]
    np.reshape(results, (len(results),1))
    return results

def getRandomModelAllocations(secs,minItems=2, maxItems=3):
    maxItems=len(secs) if maxItems > len(secs) else maxItems
    model_stocks=random.sample(secs,np.random.randint(minItems,maxItems))
    model_alloc=constrained_sum_sample_pos(len(model_stocks),100)
    #esults =
    results = dict(zip(model_stocks, model_alloc))
    return results



secs = ''.join(random.choice('ABCDE') for _ in range(4))

a = getRandomModelAllocations(secs,2,5)
print a.keys()