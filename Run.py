import dataset

import tensorflow as tf

from BaseUNet import UNet
from BaseUNetAtt import AttUNet

if __name__ == '__main__':
    img_dataset = tf.data.Dataset.from_generator(dataset.gen_dataset, 
                                            (tf.float32, tf.float32), 
                                            (tf.TensorShape([128, 800, 3]),
                                                tf.TensorShape([128, 800, 4])))
    img_dataset = img_dataset.batch(2)
    
    model = UNet()
    loss_obj = tf.keras.losses.LogCosh()
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
    optimizer = tf.keras.optimizers.Adam()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/', histogram_freq=1)
    
    model.compile(optimizer=optimizer,
              loss = loss_obj,
              metrics=[dataset.dice])
    
    history = model.fit(img_dataset,
                        epochs=30,
                        callbacks=[early_stop_cb, tensorboard_cb])
    
    model.save('./model_weights_128_800', save_format='tf')