import tensorflow as tf

from models.BaseUNetEncoder import UNetEncoder
from models.BaseUNetDecoder import UNetDecoder

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
    
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()

    def call(self, x):
        x, enc1, enc2, enc3, enc4 = self.encoder(x)
        x = self.decoder(x, enc1, enc2, enc3, enc4)
        return x