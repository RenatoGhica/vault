from datetime import datetime
from keras import backend as K

class FT_Utils:

    def __init__(self):
        pass
    
    

    @staticmethod
    def isUsingTensorFlow():
        results=(K.backend().lower() in 'tensorflow')
        return results