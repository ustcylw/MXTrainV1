# coding: utf-8
import os, sys
import tensorflow as tf


def Train(cfg, dataset, train_brick, train_callback):
    for epoch in range(cfg.MAX_EPOCH):
        for step, (x, y) in enumerate(dataset.train_data):
            train_brick.epoch = epoch
            train_brick.step = step
            try:
                with tf.GradientTape() as tape:
                    ## forward
                    predictions = train_brick.model(x)
                    ## compute loss
                    # y_onehot = tf.one_hot(y, depth=10)
                    # loss = tf.losses.categorical_crossentropy(y_onehot, predictions, from_logits=True)
                    loss = tf.losses.categorical_crossentropy(y, predictions, from_logits=False)
                    loss = tf.reduce_mean(loss)
                ## compute gradient
                grads = tape.gradient(loss, train_brick.model.trainable_variables)
                ## apply gradient
                train_brick.optimizer.apply_gradients(zip(grads, train_brick.model.trainable_variables))
                ## callback infos
                if train_brick.metric['train_loss'] is not None:
                    train_brick.metric['train_loss'].update(loss)
                if train_brick.metric['train_accuracy'] is not None:
                    train_brick.metric['train_accuracy'].update(y, predictions)
                ## calling callback
                if step % cfg.log_interval == 0:
                    if train_callback is not None:
                        if train_callback.log is not None:
                            for train_callback_log in train_callback.log:
                                train_callback_log.call(cfg, dataset, train_brick)
                    else:
                        print(epoch, step, 'loss', float(loss))
            except KeyboardInterrupt:
                # logging.info('interrupted.')
                train_callback.save_checkpoint.call(cfg, dataset, train_brick)
                # train_brick.model.save_weights(cfg.ckpt_path.format(epoch=epoch))
                # logging.info('model saved into: {}'.format(ckpt_path.format(epoch=epoch)))
                exit(0)
        ## update epoch infos
        total_num = 0
        total_correct = 0
        ## save epoch-checkpoints
        train_callback.save_checkpoint.call(cfg, dataset, train_brick)
        ## evaluate per-epoch
        for x, y in dataset.test_data:
            logits = train_brick.model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        print(epoch, 'acc:', acc)
