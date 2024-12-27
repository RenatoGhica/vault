import pandas as pd
import numpy as np
from datetime import datetime
from keras.layers import  Maximum, Average, Subtract, Add, Multiply
from keras.layers import UpSampling1D, merge, Input, LeakyReLU, Embedding, Concatenate, Dot, Reshape, Lambda, Dense, LSTM, Dropout, Activation, TimeDistributed, Flatten, RepeatVector, BatchNormalization, Conv1D, Conv2D, LocallyConnected1D

from keras.layers.pooling import GlobalAveragePooling1D, GlobalAveragePooling2D, AveragePooling1D, MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling2D
from keras.models import Sequential, Model
import keras.initializers
import keras.layers.convolutional_recurrent
import keras.initializers

from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
import keras.layers.advanced_activations


from keras import backend as K
from FT_Model import FT_Model
from FT_DataFactory import FT_DataFactory
from FT_Utils import FT_Utils

from LayerNorm1D import LayerNorm1D
from LayerFFT1D import LayerFFT1D

# =============================================================================
#             
# =============================================================================
class FT_CNN_Model(FT_Model) :
    def __init__(self,theModelName, theColGroups,theTimesteps, theFeatures, theBatchsize=1,theDataFactory=None):
        
        FT_Model.__init__(self,theModelName=theModelName)
        self.itsTimesteps=theTimesteps
        self.itsFeatures=theFeatures
        self.itsBatchsize=theBatchsize        
        #self.itsInputshape=Input(shape=(self.itsTimesteps,self.itsFeatures),batch_shape=(self.itsBatchsize,self.itsTimesteps,self.itsFeatures))
        #self.itsInputshape=Input(shape=(self.itsTimesteps,self.itsFeatures),batch_shape=(self.itsBatchsize,self.itsTimesteps,self.itsFeatures))
        self.itsInputshape=Input(shape=(self.itsTimesteps,self.itsFeatures))
        

        self.itscolGroups=theColGroups
        self.itsCategoryCount=len(self.itsTransformer.itsReturnBuckets) + 1
        self.itsDataFactory=theDataFactory   
        self.itsModel=self._create()

        
        
        
    def get(self):
        return self.itsModel
# =============================================================================


    def _create9999(self):
        
        i = self.itsInputshape
        n_cat=self.itsCategoryCount
        l=self.itsTimesteps
        
        
        num_dims=self.itsFeatures
        theModels=list()
        
        
        b_init, pad,k,s='ones','same',3,1
        
        kr,ar = None, None
        

        theAxis=-1
        
        W=8
        colGroups = ['high']
        colGroups = ['high.005','high.020','high.050','high.100']
        
# =============================================================================
# 2 D !!            
# =============================================================================
        
        #K.set_image_dim_ordering('th')
        
        #print i._keras_shape
        
        isBackend_TF=FT_Utils.isUsingTensorFlow()
        x = Reshape((self.itsTimesteps,self.itsFeatures,1))(i)
        #isBackend_TF=(K.backend().lower() in 'tensorflow')
        for g in colGroups:
            group = self.itsDataFactory.getColumnsIndex(g)
            L=len(group)            
            if (L==0):
                continue

            #x = Reshape((self.itsTimesteps,self.itsFeatures,1))(i)

            # TF workaround for complex slicing
            if (isBackend_TF):
            
                inner_list = list()
                for item in group:
                    x1 = Lambda(lambda a: a[:,:,item,:],output_shape=(self.itsTimesteps,1,1))(x)
                    inner_list.append(x1)
    
                if len(inner_list) > 1:
                    x1 = Concatenate(axis=-2)(inner_list)            
                else:
                    x1 = inner_list[0]            
                s=x1._keras_shape
                x1 = Reshape((s[1],s[2],s[3]))(x1)

            else:
                x1 = Lambda(lambda a: a[:,:,group,:],output_shape=(self.itsTimesteps,L,1))(x)
                
            
            
            
            x = Conv2D(W,(3,3),padding=pad,data_format='channels_last')(x1)            
            #x = BatchNormalization(axis=theAxis)(x)            
            x = Activation('relu')(x)
            x = MaxPooling2D((2,2),padding=pad)(x)    

            x = Conv2D(W*3,(3,3),padding=pad,data_format='channels_last')(x)
            #x = BatchNormalization(axis=theAxis)(x)            
            x = Activation('relu')(x)
            x = MaxPooling2D((2,2),padding=pad)(x)    
            
            x = Conv2D(W,(3,3),padding=pad,data_format='channels_last')(x)
            #x = BatchNormalization(axis=theAxis)(x)            
            x = Activation('relu')(x)
            x = MaxPooling2D((2,2),padding=pad)(x)    
            
            #x = Conv2D(W*2,(4,1),padding=pad,data_format='channels_last')(x)
            #x = BatchNormalization(axis=theAxis)(x)            
            #x = Activation('relu')(x)
            #x = MaxPooling2D((2,1),padding=pad)(x)    
            
#            x = Conv2D(W*3,(4,1),padding=pad,data_format='channels_last')(x)
#            x = BatchNormalization(axis=theAxis)(x)            
#            x = Activation('relu')(x)
            #x = MaxPooling2D((2,2),padding=pad)(x)    

            theModels.append(x)
        
        if len(theModels) > 1:
            l = Concatenate(axis=-1)(theModels)            
        else:
            l = theModels[0]
    
        l = Conv2D(32,(3,3),padding=pad,data_format='channels_last')(l)
        #l = BatchNormalization(axis=theAxis)(l)            
        l = Activation('relu')(l)
        l = MaxPooling2D((2,2),padding=pad)(l)    
        l = Flatten()(l)  
        
        l = Dropout(0.3)(l) 
        l = Dense(128,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(l)        
        l = Activation('relu')(l)
        l = Dropout(0.3)(l)         
        l = Dense(64,bias_initializer=b_init,kernel_regularizer=kr,activity_regularizer=ar)(l)           
        l = Activation('relu')(l)
        l = Dropout(0.3)(l) 
        l = Dense(n_cat,activation='softmax')(l)
                
        theModel = Model(inputs=[i], outputs=l)
        self.itsModel=theModel
        return self.itsModel



# =============================================================================
# 
# =============================================================================
    def _create99(self):
        #exit(1)
        i = self.itsInputshape
        n_cat=self.itsCategoryCount
        l=self.itsTimesteps
        
        
        #num_dims=self.itsFeatures
        num_dims = len(self.itsTransformer.itsReturnBuckets)
        theModels=list()
        
        
        b_init, pad,k,s='ones','same',3,1
        
        kr,ar = None, None
        #kr = regularizers.l2(0.0005)
        #ar = regularizers.l2(0.0005)
        #ar = regularizers.l1(0.001)
        

        theAxis=1
        
        W=16
        #colGroups = ['#rsi','#ema','#mfi','#adx','#raw','#rand']
        #colGroups = ['#rsi','#ema','#mfi','#adx']
        
        
        #colGroups = ['#rsi.high','#rsi.low','#rsi.close']
        #5,10,15,20,30,35,40,45,50,55,60,80,100,180
        colGroups = ['high.005','high.010','high.015','high.020','high.030','high.050','high.060','high.080','high.100']
        
        #colGroups = ['003','005','008','013','021','034','055']
        #colGroups=self.itsDataFactory.getColumns()
        #downsamples=[1,2]
        
        l_reluA=0.01
        
            

        for g in colGroups:
            group = self.itsDataFactory.getColumnsIndex(g)
            L=len(group)            
            if (L==0):
                continue


            isBackend_TF=FT_Utils.isUsingTensorFlow()
            #x = Reshape((self.itsTimesteps,self.itsFeatures,1))(i)
            #isBackend_TF=(K.backend().lower() in 'tensorflow')
                # TF workaround for complex slicing
            if (isBackend_TF):
            
                inner_list = list()
                for item in group:
                    x1 = Lambda(lambda a: a[:,:,item],output_shape=(self.itsTimesteps,1))(i)
                    inner_list.append(x1)
    
                if len(inner_list) > 1:
                    x1 = Concatenate(axis=-1)(inner_list)            
                else:
                    x1 = inner_list[0]            
                s=x1._keras_shape
                x1 = Reshape((s[1],s[2]))(x1)

            else:
                x1 = Lambda(lambda a: a[:,:,group],output_shape=(self.itsTimesteps,L))(i)

                                    
            x = Conv1D(W,11,strides=1,padding=pad,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(x1)            
            x = BatchNormalization(axis=theAxis)(x)
            x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)    
            
            
            x = Conv1D(W*2,11,strides=1,padding=pad,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(x)                    
            x = BatchNormalization(axis=theAxis)(x)
            x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)    
            
            x = Conv1D(W*3,11,strides=1,padding=pad,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(x)                    
            x = BatchNormalization(axis=theAxis)(x)
            x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)    

            #x = Conv1D(W*4,1,strides=s,padding=pad,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(x)                    
            #x = BatchNormalization(axis=theAxis)(x)
            #x = LeakyReLU(alpha=l_reluA)(x)
            
            
            theModels.append(x)
        
        if len(theModels) > 1:
            l = Concatenate(axis=1)(theModels)
        else:
            l = theModels[0]
    
        l = Conv1D(128,1,strides=1,padding=pad,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(l)                            
        l = BatchNormalization(axis=theAxis)(l)
        #l = LeakyReLU(alpha=l_reluA)(l)
        l = Activation('relu')(l)
        
        l = MaxPooling1D(2,2)(l)            

        l = Flatten()(l)  

        l = Dropout(0.3)(l) 
          
        
        l = Dense(256,bias_initializer=b_init, kernel_regularizer=kr,activity_regularizer=ar)(l)        
        l = BatchNormalization(axis=theAxis)(l)
        l = Activation('relu')(l)
                        
        l = Dropout(0.4)(l) 
        
        l = Dense(256,bias_initializer=b_init,kernel_regularizer=kr,activity_regularizer=ar)(l)           
        l = BatchNormalization(axis=theAxis)(l)
        l = Activation('relu')(l)
                        

        l = Dropout(0.3)(l) 
        l = Dense(self.itsCategoryCount,activation='softmax')(l)
        
                
        theModel = Model(inputs=[i], outputs=l)
        self.itsModel=theModel
        return self.itsModel
# =============================================================================
# backup
# =============================================================================

    def split_it(self,x,theGroups):
        return x[:,:,theGroups]
        

    
        
        
    def _create99999(self):
        
        i = self.itsInputshape
        n_cat=self.itsCategoryCount
        l=self.itsTimesteps
        
        
        #num_dims=self.itsFeatures
        num_dims = len(self.itsTransformer.itsReturnBuckets)
        theModels=list()
        
        
        b_init, pad,s='ones','valid',1
        
        kr,ar = None, None
        #kr = regularizers.l2(0.01)
        #ar = regularizers.l2(0.01)
        #ar = regularizers.l1(0.001)
        #ki=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=64)
        #ki=keras.initializers.he_uniform()
        ki=keras.initializers.he_normal()
        #ki=keras.initializers.glorot_normal()
        
# =============================================================================
#  FROZEN PARAMS
# =============================================================================
        l_reluA=0.1
        bn_scale=False 
# =============================================================================
# 
# =============================================================================
        theAxis=1
        isBackend_TF=FT_Utils.isUsingTensorFlow()
        #colGroups = ['ema.high.005','ema.high.020','ema.high.050','ema..high.100']
        #colGroups = ['rsi.high.005','rsi.high.020','rsi.high.050','rsi.high.100']
        #colGroups=self.itsDataFactory.getColumns()
        #colGroups = ['rsi.high','ema.high','value.high']
        #colGroups = ['rsi.high','ema.high','rsi.low','ema.low','mfi','adx','di','dm']
        #colGroups = ['rsi.high','ema.high','rsi.low','ema.low','mfi','raw',]
        #colGroups = ['rsi.high','ema.high','rsi.low','ema.low','mfi','raw','adx','di']
        #colGroups = ['rsi.high','ema.high','std.high','max.high','min.high','rsi.low','ema.low','std.low','max.low','min.low']\
        
        #colGroups = ['rsi.high','ema.high','mfi','angle.high','slope.high','raw.high']
        #colGroups = ['rsi.high','ema.high'] : 40 epochs: 67.5
        #colGroups = ['rsi.high','ema.high','mfi','angle.high','raw_value.high']
        #colGroups = ['rsi.high','rsi.low']
        colGroups = self.itsFeatureGroups
        #'#adx','#+di','#-di','#+dm','#-dm'
        #colGroups = ['rsi.high','mfi']
        #colGroups = ['ema.high.005','ema.high.020','ema.high.050','ema.high.100','rsi.high.005','rsi.high.020','rsi.high.050','rsi.high.100','mfi.ohlc.005','mfi.ohlc.020','mfi.ohlc.050','mfi.ohlc.100']
# =============================================================================
        for g in colGroups:
            group = self.itsDataFactory.getColumnsIndex(g)
            L=len(group)
            if (L==0):
                continue

            if (isBackend_TF): #tensorflow
                inner_list = list()
                for item in group:
                    x0 = Lambda(lambda a: a[:,:,item],output_shape=(self.itsTimesteps,1))(i)
                    inner_list.append(x0)    
                
                x0 = (Concatenate(axis=-1)(inner_list)) if (len(inner_list) > 1) else (inner_list[0])
                s=x0._keras_shape
                x0= Reshape((s[1],s[2]))(x0)
            else:  #theano
                x0 = Lambda(lambda a: a[:,:,group],output_shape=(self.itsTimesteps,L))(i)
                #x0 = Lambda(self.split_it,arguments={'theGroups':group}, output_shape=(self.itsTimesteps,L))(i)
            
            
            
            print x0._keras_shape
            x = Flatten()(x0)            
            print x._keras_shape
            mean=K.mean(x,keepdims=True)
            std=K.std(x,keepdims=True)
            x = (x - mean) / std
            #x = BatchNormalization(scale=bn_scale)(x)
            x = Reshape((self.itsTimesteps,L))(x)
            
            #x = BatchNormalization(axis=theAxis)(x)            
                
            
            # BEGIN INNER
            W,k=32,3


            x = Conv1D(16,3,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            #x = Dropout(0.5)(x)
            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)            
            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)
            x = Dropout(0.5)(x)
            

            x = Conv1D(16,5,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            #x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)
            x = Dropout(0.5)(x)
            
            x = Conv1D(16,11,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            #x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            #x = MaxPooling1D(2)(x)
            #x = Dropout(0.5)(x) 

#            x = Conv1D(W,k,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
#            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
#            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
#            
#            x = Conv1D(W,k,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
#            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
#            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)

            #x = LSTM(128,return_sequences=False,activation='rely')(x)
            #x = LSTM(128,return_sequences=True,activation='relu')(x)
            #x = LSTM(128,return_sequences=False,activation='relu')(x)

            #x = keras.layers.Dot(axes=1)([x,i])
            
            
            theModels.append(x)
        
        l = Concatenate(axis=2)(theModels) if  (len(theModels) > 1) else theModels[0]
        
        
        
        l = Conv1D(64,5,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(l)                            
        l = LeakyReLU(alpha=l_reluA)(l) #l = Activation('relu')(l)


        l = Flatten()(l)  
        
        l = Dropout(0.5)(l) 
                 
        l = Dense(512,bias_initializer=b_init,kernel_initializer=ki)(l)   ##512 from 256     
        l = LeakyReLU(alpha=l_reluA)(l) #l = Activation('relu')(l)        
        l = Dropout(0.5)(l) 
        l = Dense(256,bias_initializer=b_init,kernel_initializer=ki)(l)           
        l = LeakyReLU(alpha=l_reluA)(l) #l = Activation('relu')(l)
        
        l = Dense(self.itsCategoryCount,activation='softmax')(l)
        
                
        theModel = Model(inputs=[i], outputs=l)
        
        self.itsModel=theModel
        return self.itsModel
# =============================================================================
# 
# =============================================================================
    def _create(self):
        
        i = self.itsInputshape
        n_cat=self.itsCategoryCount
        l=self.itsTimesteps
        
        
        #num_dims=self.itsFeatures
        num_dims = len(self.itsTransformer.itsReturnBuckets)
        theModels=list()
        
        
        b_init, pad,s='ones','same',1
        
        kr,ar = None, None
        #kr = regularizers.l2(0.001)
        #ar = regularizers.l2(0.001)
        #ar = regularizers.l1(0.001)
        #ki=keras.initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=64)
        #ki=keras.initializers.he_uniform()
        ki=keras.initializers.he_normal()
        #ki=keras.initializers.glorot_normal()
        
        theAxis=1
        #colGroups = ['rsi.high','ema.high']
        colGroups = self.itscolGroups
        #'#adx','#+di','#-di','#+dm','#-dm'
        #colGroups = ['rsi.high','mfi']
        #colGroups = ['ema.high.005','ema.high.020','ema.high.050','ema.high.100','rsi.high.005','rsi.high.020','rsi.high.050','rsi.high.100','mfi.ohlc.005','mfi.ohlc.020','mfi.ohlc.050','mfi.ohlc.100']


        isBackend_TF=FT_Utils.isUsingTensorFlow()
        

# =============================================================================
#  FROZEN PARAMS
# =============================================================================
        l_reluA=0.1
        bn_scale=False 
# =============================================================================
# 
# =============================================================================
        for g in colGroups:
            group = self.itsDataFactory.getColumnsIndex(g)
            L=len(group)
            if (L==0):
                continue

            if (isBackend_TF): #tensorflow
                inner_list = list()
                for item in group:
                    x0 = Lambda(lambda a: a[:,:,item],output_shape=(self.itsTimesteps,1))(i)
                    inner_list.append(x0)    
                
                x0 = (Concatenate(axis=-1)(inner_list)) if (len(inner_list) > 1) else (inner_list[0])
                s=x0._keras_shape
                x0= Reshape((s[1],s[2]))(x0)
            else:  #theano
                x0 = Lambda(lambda a: a[:,:,group],output_shape=(self.itsTimesteps,L))(i)
                #x0 = Lambda(self.split_it,arguments={'theGroups':group}, output_shape=(self.itsTimesteps,L))(i)
            
            
            
            #print 'before:',x0._keras_shape
            x = Flatten()(x0)
            x = LayerNorm1D()(x)
            x = Reshape((self.itsTimesteps,L))(x)
            #x = BatchNormalization(axis=theAxis)(x)            


            x = Conv1D(32,3,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)            
            #x = Conv1D(64,5,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            #x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
            #x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            
            #x = Conv1D(128,11,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
            #x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
            #x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
            #x = MaxPooling1D()(x)
            #x = Dropout(0.5)(x) 

#            x = Conv1D(W,k,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
#            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
#            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)
#            
#            x = Conv1D(W,k,strides=1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(x)           
#            x = BatchNormalization(axis=theAxis,scale=bn_scale)(x)
#            x = LeakyReLU(alpha=l_reluA)(x) #x = Activation('relu')(x)

            #x = LSTM(128,return_sequences=False,activation='rely')(x)
            #x = LSTM(128,return_sequences=True,activation='relu')(x)
            #x = LSTM(128,return_sequences=False,activation='relu')(x)

            #x = keras.layers.Dot(axes=1)([x,i])
            
            
               
            theModels.append(x)
        
        l = Concatenate(axis=2)(theModels) if  (len(theModels) > 1) else theModels[0]
        
        
        l = Dropout(0.5)(l)
        d=l._keras_shape[2]
        
        #print d
        l = Conv1D(d*2,1,padding=pad,bias_initializer=b_init,kernel_initializer=ki)(l)                            
        l = LeakyReLU(alpha=l_reluA)(l) #l = Activation('relu')(l)
        l = BatchNormalization(axis=theAxis,scale=bn_scale)(l)
        
        l = Dropout(0.5)(l)
        #l = MaxPooling1D()(l)
        #l = GlobalAveragePooling1D()(l)
#        l = Dropout(0.5)(l)
#        
        l = Flatten()(l)  
#
##        
        l = Dense(128,bias_initializer=b_init,kernel_initializer=ki)(l)           
        l = LeakyReLU(alpha=l_reluA)(l) 

        l = Dropout(0.3)(l) 

        l = Dense(128,bias_initializer=b_init,kernel_initializer=ki)(l)           
        l = LeakyReLU(alpha=l_reluA)(l) 

        
        l = Dense(self.itsCategoryCount,activation='softmax')(l)
        
                
        theModel = Model(inputs=[i], outputs=l)
        
        self.itsModel=theModel
        return self.itsModel

        
        
def K_localNormalizeMinMax(x):
    return x
#    orig_shape=x.shape    
#    print orig_shape
#    v = x.flatten()         
#    # rescale to -1..1 while keeping all relationshipts between features in the same group (ie RSI)
#    results = ((v - np.nanmin(v,axis=ax)) / (np.nanmax(v,axis=ax)-np.nanmin(v,axis=ax))) * 2 - 1
#    results = np.reshape(results,orig_shape)
#    return results

#    def min_max_pool1d(self,x):
#        max_x =  K.pool2d(x, pool_size=2, strides=1)
#        min_x = -K.pool2d(-x, pool_size=2, strides=1)
#        return K.concatenate([max_x, min_x], axis=1) # concatenate on channel
#    
#    def min_max_pool1d_output_shape(self,input_shape):
#        shape = list(input_shape)
#        shape[1] *= 2
#        return tuple(shape)