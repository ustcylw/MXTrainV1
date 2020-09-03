# coding: utf-8
import os, sys
import tensorflow as tf


def Train(cfg, dataset, train_brick):
    # model = tf.keras.applications.MobileNetV2(input_shape=(target_size, target_size, 3), weights=None, include_top=True, classes=5)

    # todo: why keras fit converge faster than tf loop?
    train_brick.model.compile(
        optimizer=train_brick.optimizer,  # 'adam',
        loss=train_brick.loss,  # 'sparse_categorical_crossentropy',
        metrics=train_brick.metric  # ['accuracy']
    )
    try:
        train_brick.model.fit(
            dataset.train_dataset,
            epochs=50,
            steps_per_epoch=700
        )
    except KeyboardInterrupt:
        model.save_weights(ckpt_path.format(epoch=0))
        logging.info('keras model saved.')
    model.save_weights(ckpt_path.format(epoch=0))
    model.save(os.path.join(os.path.dirname(ckpt_path), 'flowers_mobilenetv2.h5'))





