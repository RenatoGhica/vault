from FT_DataFactory import FT_DataFactory
from FT_Transformer import FT_Transformer, colType
from FT_Charts import FT_Charts
import pandas as pd
from pandas import DataFrame
from pandas import concat


import numpy as np


from keras import optimizers, losses
from keras.layers import Dense, LSTM, Dropout, Activation, TimeDistributed, Flatten, RepeatVector
from keras.models import Sequential, model_from_json
from keras.utils.vis_utils import plot_model
from keras.utils import plot_model
from keras.layers import Input, Concatenate, Dot, Reshape, Average, Lambda, Dense, LSTM, Dropout, Activation, TimeDistributed, Flatten, RepeatVector, BatchNormalization, Conv1D, Conv2D
from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D, MaxPooling1D
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
import keras.utils

from sklearn.utils import class_weight


from recurrentshop import LSTMCell, RecurrentSequential
#from .cells import LSTMDecoderCell, AttentionDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input
from keras.callbacks import TensorBoard
from time import time

from keras import backend as K

import scipy
from pathlib import Path

class FT_Model:
    
        colsStandard=['volume','open','high','low','close']  
        colsStandardID=['ticker','date']  
        colSeparator='|'
        colsValidType=[colType.BOOLEAN.value, colType.PCT.value, colType.UNITLESS.value]
        
        def __init__(self, theModelName='model'):
            self.itsModelName=theModelName;
            self.itsTransformer=FT_Transformer()
            self.itsModel=None
            self.itsModelFileName=self.itsModelName+'.json'
            self.itsModelWeightsFileName=self.itsModelName+'.h5'
            self.itsModelVisualFileName=self.itsModelName+'.jpg'
           
        def load_weights(self):            
            self.itsModel.load_weights(self.itsModelWeightsFileName)
            self.itsModel.reset_states()
            print 'Loaded weights from disk : [{}]'.format(self.itsModelWeightsFileName)
            print self.itsModel.summary()
            return self.itsModel

        def load(self):            
            f_name=self.itsModelFileName           
            json_file = open(f_name, 'r')
            model = json_file.read()
            json_file.close()
            self.itsModel=model_from_json(model)
            self.itsModel.load_weights(self.itsModelWeightsFileName)
            print 'Loaded model from disk : [{}]'.format(self.itsModelFileName)
            print self.itsModel.summary()
            return self.itsModel
        
        def save(self):            
            mdl=self.itsModel            
            model_json = mdl.to_json()
            with open(self.itsModelFileName, "w") as json_file:
                json_file.write(model_json)
            mdl.save_weights(self.itsModelWeightsFileName)
            
            #plot_model(mdl, to_file=self.itsModelVisualFileName)
            plot_model(mdl,to_file=self.itsModelVisualFileName,show_shapes=True)
            
            #SVG(model_to_dot(mdl).create(prog=self.itsModelVisualFileName, format='jpg'))
            print 'Saved model to disk : [{},{},{}]'.format(self.itsModelFileName, self.itsModelWeightsFileName,self.itsModelVisualFileName)
            return self.itsModelFileName, self.itsModelWeightsFileName

        def getModel(self):
            return self.itsModel
# =============================================================================
#         
# =============================================================================
        def getXY(self,theData, theSplit=0.2, thePredictions=1, theBatchSize=100,isCategorical=False):    
            #TODO : optimize
            g=np.asarray(theData)
            if (len(g)==0):
                print 'no data returned frmo getXY: {}'.format(len(g))
            theTimesteps=g.shape[1]
            all_X, all_Y = g[:,:, :-thePredictions], g[:,-1, -thePredictions]           
            
            l=len(all_X)
            lb= l - (l%theBatchSize)
            all_X=all_X[-lb:]
            all_Y=all_Y[-lb:]
            
            l-=l % theTimesteps
            L=  int(l/theTimesteps)
            L=int(L*theSplit)*theTimesteps
            if L==0: L=theTimesteps
        
            train_size=l-L
            train_size-=train_size % theBatchSize
            
            X=all_X[0:train_size]
            Y=all_Y[0:train_size]
            
            test_X=all_X[train_size:]
            test_Y=all_Y[train_size:]

            if (isCategorical):
                Y_cat=keras.utils.to_categorical(Y,12)
                test_Y_cat=keras.utils.to_categorical(test_Y,12)
               
                
            return X, Y, test_X, test_Y
# =============================================================================
# 
# =============================================================================

            
        def compileModel(self, theModelData,  theLossFn='binary_crossentropy', theOptimizerFn='rmsprop',theMetrics=['accuracy'], thePredictions=1):            

            m=self.itsModel
            self.save()
            #print theModelData.shape
            #model_data=np.asarray(theModelData)
            samples, timesteps, features =theModelData.shape
            features -= thePredictions
            #mdl_cells=features+1
            
            print '[compileModel] samples: {}, timesteps: {}, features:{}'.format(samples, timesteps,features)
            m.compile(loss=theLossFn, optimizer=theOptimizerFn,metrics=theMetrics)             
            print  m.summary()
            
            
            return self.itsModel

        def compileCategoricalModel(self, theModelData, theCategories=2,thePredictions=1):            
            theLossFn='sparse_categorical_crossentropy'
            theMetrics=['sparse_categorical_accuracy']
            theOptimizer=optimizers.adam(0.001)
            
            return self.compileModel(theModelData,theLossFn,theOptimizer, theMetrics,thePredictions)
# =============================================================================
# 
# =============================================================================
        def normalizeMinMax(self,x,ax=1):

            results = (x - np.nanmin(x,axis=ax)) / (np.nanmax(x,axis=ax)-np.nanmin(x,axis=ax))
            #results = (x - np.nanmax(x,axis=ax) + (x - np.nanmin(x,axis=ax) )) / (np.nanmax(x,axis=ax)-np.nanmin(x,axis=ax))
            results = results * 2 - 1
            return results

        def runModel(self,theData, thePredictions=1, theSplit=0.1,theWindowSize=30,theBatchSize=1,theEpochs=1,isNormalizationEnabled=False):
            m = self.itsModel
            reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.1,patience=10, min_lr=0.00001) 
            early_stop = EarlyStopping(monitor='sparse_categorical_accuracy', min_delta=0.0001, patience=20, verbose=1, mode='auto')
            
            tensorboard = (lambda x: x)
            
            print '----using backend : {}'.format(K.backend())
            
            isBackend_TF=(K.backend().lower() in 'tensorflow')
#            
            itsCallbacks=[reduce_lr,early_stop]
            itsCallbacks=[early_stop]
            #itsCallbacks=[]
           
            if (isBackend_TF):
                tensorboard = TensorBoard(log_dir="logs/{}".format(time()), histogram_freq=1,write_grads=False,write_graph=True, write_images=True)
                tensorboard.set_model(m)
                itsCallbacks.append(tensorboard)
            
        
            X, Y, test_X, test_Y = self.getXY(theData,theSplit,thePredictions,theBatchSize,isCategorical=False)
            print X.shape, Y.shape
            print test_X.shape, test_Y.shape
            
            
            #print 'test_y',test_Y[0:1000]
            
            if (isNormalizationEnabled):
                print '****normalizing - not implemented...'
                
                
            cw_train = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
            sw_train = class_weight.compute_sample_weight('balanced', Y)
            print 'class weights:',cw_train
            
            #tensorboard.set_model(m)
            #m.fit(X, Y, epochs=theEpochs, batch_size=theBatchSize, shuffle=False,verbose=1, validation_split=theSplit 
            m.fit(X, Y, epochs=theEpochs, batch_size=theBatchSize, shuffle=True,verbose=1, validation_data=(test_X,test_Y)
            ,callbacks=itsCallbacks
            #,class_weight=cw_train
            #,sample_weight=sw_train
            )
            scores=m.evaluate(test_X,test_Y,batch_size=theBatchSize)
            
            print("\n\r %s: %.2f%%" % (m.metrics_names[1], scores[1]*100))
            return scores

# =============================================================================
 
# 
# =============================================================================

        def predict(self,X,theBatchSize=1):
            return self.itsModel.predict(X)

        def evaluate(self,theX, theY,theBatchSize=1):
            #cw_test = class_weight.compute_class_weight('balanced', np.unique(theY), theY)
            sw_test = class_weight.compute_sample_weight('balanced', theY)

            #return self.itsModel.evaluate(theX, theY, theBatchSize,sample_weight=sw_test,verbose=1)
            return self.itsModel.evaluate(theX, theY, theBatchSize,verbose=1)

# =============================================================================
#         
# =============================================================================
        def featureSummary(self,inFile='../data/modeldata_stage2.csv',outFile='../data/feature_stats.csv'):
            df=pd.read_csv(inFile)
            
            #print df.head()
            stats=pd.DataFrame(columns=['f_name','f_type','f_count','f_min','f_max','invariant','isScalar'])
            

            for colName in df.columns:
                

                c=df[colName]
                #check if all values in a column are the same
                val=df[colName][0]
                isInvariant=all(v==val for v in c)
                isScalar=(c.max()>100)
                minV=c.min()
                maxV=c.max()
                cnt=c.count()
                
                          
                if (isInvariant):
                    print '[should delete] ',colName,' has zero variation in values [',val,'] for',c.count(),' values'
                    

                if (isScalar):
                    print '[should delete] ',colName,' has scalar range [',minV,'..',maxV,']'

    
                stats.loc[len(stats)]=[colName,c.dtype,cnt,minV,maxV,isInvariant,isScalar]     
                
                
            stats.head()
            stats.to_csv(outFile)      
            print '[info]',len(stats), ' features created!'
# =============================================================================
#            
# =============================================================================
        def normalize(self,inFile='../data/modeldata_stage1.csv',outFile='../data/modeldata_stage2.csv'):
            df=pd.read_csv(inFile)
            df=df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

            print df.shape
            df= df.drop(df[df.high <=0].index)
            print df.shape
            df= df.drop(df[df.low <=0].index)
            print df.shape
            df= df.drop(df[df.close <=0].index)
            print df.shape
            print '[done]'

            
            validCols=self.colsValidType
            sel= [col for col in df.columns if  any(x in col for x in validCols) ]
            cols= self.colsStandardID + self.colsStandard + sorted(sel)
            
            df=df[cols]
            
            for colName in sel:
                
                print 'processing: ',colName
                c=df[colName]
                #val=df[colName][0]
                minV=c.min()
                maxV=c.max()
                

                isInvariant=all(minV==maxV for v in c)
                isScalar=(c.max()>100)
                minV=c.min()
                maxV=c.max()
                #cnt=c.count()
                
                          
                if (isInvariant or isScalar):
                    print '[',colName,'] invariant=',isInvariant,'; scalar=',isScalar
                    #df.drop(colName,axis=1,inplace=True)
                    
            df.to_csv(outFile,index=False)
           
            #for colName in df.columns:
# =============================================================================
#                 
# =============================================================================
        def target(self,target='#return.close.001.pct', theShift=0,theTargets=1, inFile='../data/modeldata_stage2.csv',outFile='../data/modeldata_stage3.csv'):
            
            thePredictions=5
            
            colRowID=['row']
            colTARGETID=list()
            theGroups=list()
            
            df=pd.read_csv(inFile)
            ##results=results.round(3)
            g=df.groupby(by=['ticker'])
            
            
            for i in range (0,theTargets):
                colName='TARGET'+str(i)
                colTARGETID.append(colName)
                
            for name, group in g:
                l=len(group)
                group['row']=[x for x in range(1,l+1)]
                nextPred=theShift
                for col in colTARGETID:
                    group[col]=(group[target].shift(-nextPred))
                    nextPred+=1
                theGroups.append(group)
            
            df=pd.concat(theGroups)
            valid_cols=self.colsValidType
            sel_cols= [col for col in df.columns if  any(x in col for x in valid_cols) ]
            #cols=['ticker','date','TARGET'] + sorted(sel)
            return_cols=[col for col in df.columns if '#return' in col]
            return_cols=np.unique(return_cols)
            
            zscore_cols=[col for col in df.columns if '#zscore' in col]
            zscore_cols=np.unique(zscore_cols)

            model_cols=set(sel_cols) - set (return_cols) - set(zscore_cols)
            df[self.colSeparator]=''
            #cols= self.colsStandardID + colRowID + self.colsStandard + sorted(return_cols) + list(self.colSeparator) + sorted(zscore_cols) + sorted(model_cols)  + sorted(colTARGETID)
            cols= self.colsStandardID + colRowID + self.colsStandard + sorted(return_cols) + sorted(zscore_cols) + list(self.colSeparator) +  sorted(model_cols)  + sorted(colTARGETID)
            
            #df=df[cols].round(4)
            #results=df.reindex_axis(cols,axis=1)
            #df['TARGET0']=df['TARGET0']
            results=df[cols]
            
            results.dropna(inplace=True,axis=0)
            results.to_csv(outFile,index=False)  
            return results

# =============================================================================
# 
# =============================================================================
def main():
            f_stage1    ='../data/modeldata_stage1.csv'
            f_stage2    ='../data/modeldata_stage2.csv'
            f_stage3    ='../data/modeldata_stage3.csv'
            f_stats     ='../data/modeldata_mystats.csv'
                
            m=FT_Model()
            m.normalize(inFile=f_stage1,outFile=f_stage2)
            m.featureSummary(inFile=f_stage2,outFile=f_stats)
            #m.target('#return.close.005.pct',-5,inFile=f_stage2,outFile=f_stage3)    
            m.target('#zscore.close.020.unitless',-1,inFile=f_stage2,outFile=f_stage3)    
            
           

if __name__ == "__main__":
    main()            