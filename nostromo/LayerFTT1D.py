from keras import backend as K
from keras.engine.topology import Layer
import keras.initializers
import theano.tensor.fft



class LayerFFT1D(Layer):
    def __init__(self, output_dim, **kwargs):
        super(LayerFFT1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.trainable_weights = []
        super(LayerFFT1D,self).build(input_shape)

    def call(self, x, mask=None):
        #return K.tf.complex_abs(K.tf.fft(K.tf.complex(x, K.tf.zeros_like(x))))
        results = theano.tensor.fft.rfft(x)
        return results
        

    def get_output_shape_for(self, input_shape):
        return (input_shape)
# =============================================================================
#     
# =============================================================================
    
