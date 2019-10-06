import dataset
import DataGen
import data

import tensorflow as tf

from models.BaseUNet import UNet
from models.BaseUNetAtt import AttUNet

if __name__ == '__main__':
    # img_dataset = tf.data.Dataset.from_generator(dataset.gen_dataset,
    #                                         (tf.float32, tf.float32),
    #                                         (tf.TensorShape([128, 800, 3]),
    #                                             tf.TensorShape([128, 800, 4])))

    # img_dataset = img_dataset.batch(2)

    img_dataset = DataGen.DataGenerator(data.preproc_train_csv, data.train_dir)
    
    model = UNet()
    loss_obj = tf.keras.losses.BinaryCrossentropy()
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=4)
    accuracy_metric = tf.keras.metrics.Accuracy()
    optimizer = tf.keras.optimizers.Adam()
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='./log_dir/', histogram_freq=1)
    
    model.compile(optimizer=optimizer,
              loss = loss_obj,
              metrics=[dataset.dice, accuracy_metric, iou_metric])
    
    history = model.fit_generator(img_dataset,
                        epochs=3,
                        callbacks=[early_stop_cb, tensorboard_cb])
    
    model.save('./model_weights_256_400', save_format='tf')