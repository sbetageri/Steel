import tensorflow as tf

class UNetEncoder(tf.keras.Model):
    def __init__(self):
        super(UNetEncoder, self).__init__()
        
        self.dl1_conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.dl1_conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)
        
        self.dl2_conv1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.dl2_conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        
        self.dl3_conv1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.dl3_conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        
        self.dl4_conv1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.dl4_conv2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        
        self.dl5_conv1 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')
    
    def call(self, x):
        x = self.dl1_conv1(x)
        x = self.dl1_conv2(x)
        enc1 = x
        
        x = self.maxpool(x)
        
        x = self.dl2_conv1(x)
        x = self.dl2_conv2(x)
        enc2 = x
        
        x = self.maxpool(x)
        
        x = self.dl3_conv1(x)
        x = self.dl3_conv2(x)
        enc3 = x
        
        x = self.maxpool(x)
        
        x = self.dl4_conv1(x)
        x = self.dl4_conv2(x)
        enc4 = x
        
        x = self.maxpool(x)
        
        x = self.dl5_conv1(x)
        return x, enc1, enc2, enc3, enc4