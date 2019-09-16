import dataset

import tensorflow as tf

from BaseUNet import UNet

if __name__ == '__main__':
    img_dataset = tf.data.Dataset.from_generator(dataset.gen_dataset, 
                                            (tf.float32, tf.float32), 
                                            (tf.TensorShape([256, 1600, 3]), 
                                                tf.TensorShape([256, 1600, 4])))
    img_dataset = img_dataset.batch(2)
    
    model = UNet()
    loss_obj = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/', histogram_freq=1)
    
    model.compile(optimizer=optimizer,
              loss = loss_obj,
              metrics=[dataset.dice])
    
    history = model.fit(img_dataset,
                        epochs=10,
                        callbacks=[early_stop_cb, tensorboard_cb])