from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, logging, pickle, importlib, re, copy, random, tqdm
import os.path as osp
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

from utils.base_solver import BaseSolver
from utils import config as cfg
from utils import dataset, utils

from run_symnet import make_parser


def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    parser.add_argument("--test_set", type=str, default='test',
        choices=['test','val'])
    args = parser.parse_args()


    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, args.test_set, 
        batchsize=args.test_bz)
    

    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(test_dataloader, args)


    with utils.create_session() as sess:
        sw = SolverWrapper(net, test_dataloader, args)
        sw.trainval_model(sess)




################################################################################



class SolverWrapper(BaseSolver):
    def __init__(self, network, test_dataloader, args):
        logger = self.logger("init")
        self.network = network

        self.test_dataloader = test_dataloader
        

        self.args = args
        self.trained_weight = os.path.join(
            cfg.WEIGHT_ROOT_DIR, args.name, 
            'snapshot_epoch_{}.ckpt'.format(args.epoch))
        logger.info("trained weight <= "+self.trained_weight)

        self.filename = "%s_%s_ep%d.pkl"%(args.name, args.test_set, args.epoch)
        self.filepath = osp.join(cfg.ROOT_DIR, "data/obj_scores", self.filename)
        assert args.force or not os.path.exists(self.filepath), self.filepath
        logger.info("obj prediction => "+self.filepath)

            


    def initialize(self, sess):
        """weight initialization"""
        logger = self.logger('initialize')
        logger.info('Restoring whole snapshots from {:s}'.format(
            self.trained_weight))
        saver_restore = tf.train.Saver()
        saver_restore.restore(sess, self.trained_weight)




    def construct_graph(self, sess):
        logger = self.logger('construct_graph')

        with sess.graph.as_default():
            if cfg.RANDOM_SEED is not None:
                tf.set_random_seed(cfg.RANDOM_SEED)

            prob_op = self.network.build_network(test_only=True)

        return prob_op

        

    def trainval_model(self, sess):
        logger = self.logger('train_model')
        logger.info('Begin training')

        prob_op = self.construct_graph(sess)

        self.initialize(sess)
        sess.graph.finalize()


        ##################################### test czsl #######################################

        predictions = []
        obj_truth = []

        for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):
            pred = self.network.test_step_no_postprocess(sess, batch, prob_op)
            predictions.append(pred)
            obj_truth.append(batch[2])

        
        predictions = np.concatenate(predictions, axis=0)

        print("Saving %s tensor to %s"%(str(predictions.shape),self.filename))
        with open(self.filepath, 'w') as fp:
            pickle.dump(predictions.tolist(), fp)


        obj_truth = np.concatenate(obj_truth, axis=0)
        obj_pred = predictions.argmax(axis=1)
        print(obj_pred.shape, obj_truth.shape)
        
        match = (obj_pred==obj_truth).astype(float)
        acc = np.sum(match)/match.shape[0]

        print("Obj acc = %.3f"%(acc*100))


        logger.info('Finished.')




if __name__=="__main__":
    main()
