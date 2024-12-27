import pandas as pd
import numpy as np
#import pylab
from cycler import cycler
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.widgets import Cursor
import matplotlib.dates as mdates
from itertools import cycle

class FT_Charts():

     def __init__(self,theDataFrame, theTimeSeriesColumn='date',theRows=1):
        self.timeSeriesColumn=theTimeSeriesColumn
        self.df=theDataFrame
        self.X = np.transpose(self.df[self.timeSeriesColumn])
        self.X = self.X.astype(dt.datetime)
        self.fig=plt.figure()#figsize=(9,4))
        self.itsSubPlots = list()
        #self.col_gen = prop_cycle(cycler('bgrcmk'))
        self.chartPadding=[0.1,0.7,.9,.1]
        self.itsRows=theRows*100 + 10
        
        #plt.style.use('fivethirtyeight')
        self.fig.autofmt_xdate()
        #plt.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
        plt.rc('axes',prop_cycle= cycler('color',['r', 'k', 'c']))
        #pylab.ylim([0,100])
        
        
     def addSeries(self,theColumn,theGridPosition=1, isSubPlot=True,theTitle='n/a'):        
         #ax=self.fig.add_subplot(theGridPosition)
         #ax.plot(self.X,self.df[theColumn],label=theColumn,color=self.col_gen.next())
         gridLoc=self.itsRows + theGridPosition
         #print gridLoc
         self.itsSubPlots.append((self.fig.add_subplot(gridLoc),theColumn,gridLoc,theTitle))
         


     def show(self):
        
         for ax,col,grid,title in self.itsSubPlots:
             ax=self.fig.add_subplot(grid)
             plt.axis('equal')  
             box=ax.get_position()
             #color=cm(np.random.randint(0,4)//3*3.0/NUM_COLORS)
             ax.plot(self.X,self.df[col].values,label=col,c=np.random.rand(3))
             
             ax.set_title(title)
             ax.legend(loc='center left', bbox_to_anchor=(1.01,.5))                          
             ax.set_position([box.x0, box.y0,  box.width * 0.8, box.height]) 
             #pylab.ylim([0,100])
             ax.relim()
             
             ax.grid('on')
             #plt.subplots_adjust(left=self.chartPadding[0],right=self.chartPadding[1],top=self.chartPadding[2],bottom=self.chartPadding[3])
             
         #plt.axis('equal')         
         plt.autoscale(True)    
         plt.subplots_adjust(left=self.chartPadding[0],right=self.chartPadding[1],top=self.chartPadding[2],bottom=self.chartPadding[3])
         
         self.fig.autofmt_xdate()

         plt.show()
#===============================================================================
# =
def test():
     from FT_DataFactory import FT_DataFactory
     data = FT_DataFactory()
     data.loadData('../data/stocks_2016.csv',1000)
     #data.info()
     df=data.getData()
     df.groupby(['ticker'])
     df=df.loc['A']
     print np.min(df['close'])
     
     
     if (False):
         X = np.transpose(df['date'])
         X = X.astype(dt.datetime)
         fig=plt.figure(figsize=(8,4))
         
         pltMain = fig.add_subplot(311)
         boxMain=pltMain.get_position()
         plt.setp(pltMain.get_xticklabels(), visible=False)
         pltMain.plot(X,df['open'],label='open')
         pltMain.plot(X,df['close'],label='close')
         pltMain.legend(loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True)
         pltMain.set_position([boxMain.x0, boxMain.y0, boxMain.width * 0.8, boxMain.height])
         
    
# =============================================================================
#          pltSub1 = fig.add_subplot(312,sharex=pltMain)
#          boxSub1=pltSub1.get_position()     
#          plt.setp(pltSub1.get_xticklabels(), visible=False)
#          pltSub1.plot(X,df['high'],label='high')
#          pltSub1.legend(loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True)
#          pltSub1.set_position([boxSub1.x0, boxSub1.y0, boxSub1.width * 0.8, boxSub1.height])
#          
#     
#          pltSub2 = fig.add_subplot(313,sharex=pltMain)
#          boxSub2=pltSub2.get_position()     
#          pltSub2.plot(X,df['low'],label='low')
#          pltSub2.set_position([boxSub2.x0, boxSub2.y0, boxSub2.width * 0.8, boxSub2.height])
#          pltSub2.legend(loc='center left', bbox_to_anchor=(1,0.5),ncol=1, fancybox=True, shadow=True)
#          plt.setp(pltSub2.get_xticklabels(), visible=False)
#          plt.setp(pltSub2.get_xticklabels(), visible=True)
# =============================================================================
         plt.axis('equal')
         plt.show()

     
     if (True):
         df.groupby(['ticker'])
         c=FT_Charts(df.loc['A'],'date',3)
         c.addSeries('adj_close',1)
         c.addSeries('adj_high',2)
         c.addSeries('adj_low',3)
         #c.addSeries('adj_low',4)    
         df.loc['A'].head()
         
        #c.addSeries('volume',515)     
         c.show()

if __name__ == "__main__":
    test()   