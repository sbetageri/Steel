import tensorflow as tf

from BaseUNetEncoder import UNetEncoder
from BaseUNetAttDecoder import UNetAttentionDecoder

class AttUNet(tf.keras.Model):
    def __init__(self):
        super(AttUNet, self).__init__()
    
        self.encoder = UNetEncoder()
        self.decoder = UNetAttentionDecoder()

    def call(self, x):
        x, enc1, enc2, enc3, enc4 = self.encoder(x)
        x = self.decoder(x, enc1, enc2, enc3, enc4)
        return x