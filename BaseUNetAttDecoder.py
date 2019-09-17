import tensorflow as tf

class UNetAttentionDecoder(tf.keras.Model):
    def __init__(self):
        super(UNetAttentionDecoder, self).__init__()

        self.ul1_t_conv = tf.keras.layers.Conv2DTranspose(512, 2, strides=2)
        self.ul1_conv1 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.ul1_conv2 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')
        self.att_1 = tf.keras.layers.Attention()
        
        self.ul2_t_conv = tf.keras.layers.Conv2DTranspose(256, 2, strides=2)
        self.ul2_conv1 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.ul2_conv2 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')
        self.att_2 = tf.keras.layers.Attention()
        
        self.ul3_t_conv = tf.keras.layers.Conv2DTranspose(128, 2, strides=2)
        self.ul3_conv1 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.ul3_conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')
        self.att_3 = tf.keras.layers.Attention()
        
        self.ul4_t_conv = tf.keras.layers.Conv2DTranspose(64, 2, strides=2)
        self.ul4_conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.ul4_conv2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.att_4 = tf.keras.layers.Attention()
        
        self.ul4_conv3 = tf.keras.layers.Conv2D(4, 1, activation='sigmoid')
    
    def call(self, x, enc1, enc2, enc3, enc4):
    
        x = self.ul1_t_conv(x)
        enc4 = self.att_1([x, enc4])
        x = self.ul1_conv1(enc4)
        x = self.ul1_conv2(x)
        
        x = self.ul2_t_conv(x)
        enc3 = self.att_2([x, enc3])
        x = self.ul2_conv1(enc3)
        x = self.ul2_conv2(x)
        
        x = self.ul3_t_conv(x)
        enc2 = self.att_3([x, enc2])
        x = self.ul3_conv1(x)
        x = self.ul3_conv2(x)
        
        x = self.ul4_t_conv(x)
        enc1 = self.att_4([x, enc1])
        x = self.ul4_conv1(x)
        x = self.ul4_conv2(x)
        
        x = self.ul4_conv3(x)

        return x, enc4