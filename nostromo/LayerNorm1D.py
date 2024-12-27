#from keras.layers import Layer
from keras import backend as K
from keras.engine.topology import Layer
import keras.initializers

# =============================================================================
# 
# =============================================================================

class LayerNorm1D(Layer):
    def __init__(self,   **kwargs):
        #self.output_dim = output_dim
        self.eps = 1e-6
        super(LayerNorm1D,self).__init__(**kwargs)
# =============================================================================
    def build(self, input_shape):
#        self.gamma = self.add_weight(name='gamma',
#                                     shape=input_shape,
#                                     initializer=keras.initializers.Ones(),
#                                     trainable=True)
#
#        self.beta = self.add_weight(name='beta',
#                                    shape=input_shape,
#                                    initializer=keras.initializers.Zeros(),
#                                    trainable=True,)

        super(LayerNorm1D,self).build(input_shape)

# =============================================================================
#    def call(self, x):
#        mean = K.mean(x, axis=-1, keepdims=True)
#        std = K.std(x, axis=-1, keepdims=True)
#        #return self.gamma * (x - mean) / (std + self.eps) + self.beta
#        return (x - mean) / std
    def call(self,x):
        low,high=-1.0, 1.0

        #v = K.flatten(x)
        mins = K.min(x,keepdims=False)
        maxs = K.max(x,keepdims=False)
        
        results = 1.0 - ((2.0 * (maxs - x)) / 2.0)
        print K.shape(results)
        #rescaled = high - (((high - low) * (maxs - v)) / rng)
        #results = K.reshape(rescaled,K.shape(x))

        return results
# =============================================================================
    def compute_output_shape(self, input_shape):
        #return input_shape[0], self.output_dim
        return input_shape
# =============================================================================    
    