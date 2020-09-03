# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np


class RMSE(mx.metric.EvalMetric):
    def __init__(self, rmse_version=2):
        super(RMSE, self).__init__('RMSE')
        self.rmse_version = rmse_version

    def update(self, labels, preds):
        if self.rmse_version == 0:
            rmse = self.compute_nmse_v1(labels, preds)
        elif self.rmse_version == 1:
            rmse = self.compute_nmse_v2(labels, preds)
        elif self.rmse_version == 2:
            rmse = self.mseNormlized(labels, preds)
        else:
            rmse = 0.0
        return rmse

    def compute_nmse_v1(self, ground_truth, predictions):
        targets = ground_truth##.reshape((-1, 2))
        preds = predictions##.reshape((-1, 2))

        N = preds.shape[0]
        L = preds.shape[1] / 2
        rmse = np.zeros(N)

        # print(f'preds: {preds.shape}')
        for i in range(N):
            pts_pred, pts_gt = preds[i:(i+1), :], targets[i:(i+1), :]
            # print(f'pts_gt: {pts_gt.shape}')
            interocular = np.linalg.norm(pts_gt.reshape((68, 2))[36, :] - pts_gt.reshape((68, 2))[45, :])
            # print(f'tmp: {np.linalg.norm(pts_pred - pts_gt, axis=1)}  {interocular}')
            rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

        return rmse

    def compute_nmse_v2(self, ground_truth, predictions):
        targets = ground_truth.reshape((-1, 2))
        preds = predictions.reshape((-1, 2))

        interocular = np.linalg.norm(targets[36,]) - targets[45,]
        rmse = np.sum(np.linalg.norm(preds - targets, axis=1)) / interocular

        return rmse

    def mseNormlized(self, ground_truth, pred):
        ground_truth = ground_truth.reshape((-1, 2))
        pred = pred.reshape((-1, 2))
        eyeDistance = np.linalg.norm(ground_truth[36] - ground_truth[45])
        norm_mean = np.linalg.norm(pred - ground_truth, axis=1).sum()
        if eyeDistance > 0.0:
            return (norm_mean / eyeDistance)
        else:
            return 0.0






if __name__ == '__main__':
    import random
    preds = mx.nd.array(np.array([random.randint(0, 1) for i in range(2*136)]).reshape((2, 136)))
    labels = mx.nd.array(np.array([random.randint(0, 1) for i in range(2*136)]).reshape((2, 136)))

    Rmse = RMSE()
    rmse = Rmse.update(labels[0, :].asnumpy(), preds[0, :].asnumpy())
    print('rmse: {}'.format(rmse))