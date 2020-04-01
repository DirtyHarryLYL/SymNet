from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.base_solver import BaseSolver

import os, logging, importlib, re, copy, random, tqdm, argparse
import os.path as osp
import cPickle as pickle
from pprint import pprint
from datetime import datetime
import numpy as np
from collections import defaultdict


import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

from utils import config as cfg
from utils import dataset, utils
from utils.evaluator import GCZSL_Evaluator

from run_symnet_gczsl import make_parser
from run_symnet_gczsl import SolverWrapper as GCZSLSolverWrapper


def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    utils.display_args(args, logger)

    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, args.test_set,
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(test_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)




################################################################################



class SolverWrapper(GCZSLSolverWrapper):
    def __init__(self, network, test_dataloader, args):
        logger = self.logger("init")
        self.network = network
        self.test_dataloader = test_dataloader
        self.args = args

        self.trained_weight = os.path.join(cfg.WEIGHT_ROOT_DIR, args.name, "snapshot_epoch_%d.ckpt"%args.epoch)
        self.logger("init").info("pretrained model <= "+self.trained_weight)


    def construct_graph(self, sess):
        logger = self.logger('construct_graph')

        with sess.graph.as_default():
            if cfg.RANDOM_SEED is not None:
                tf.set_random_seed(cfg.RANDOM_SEED)

            _, score_op, _ = self.network.build_network()

        return score_op

        

    def trainval_model(self, sess, max_epoch):
        logger = self.logger('train_model')
        logger.info('Begin training')

        score_op = self.construct_graph(sess)
        #for x in tf.global_variables():
        #    print(x.name)

        self.initialize(sess)
        sess.graph.finalize()



        evaluator = GCZSL_Evaluator(self.test_dataloader.dataset)


        ############################## test czsl ################################
        
        all_attr_lab = []
        all_obj_lab = []
        all_pred = defaultdict(list)

        for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

            predictions = self.network.test_step(sess, batch, score_op)  # ordereddict of [score_pair, score_a, score_o]

            attr_truth, obj_truth = torch.from_numpy(batch[1]), torch.from_numpy(batch[2])
            all_attr_lab.append(attr_truth)
            all_obj_lab.append(obj_truth)

            for key in score_op.keys():
                all_pred[key].append(predictions[key][0])


        for key,value in all_pred.items():
            logger.info(key)
            report = self.test(self.args.epoch, evaluator, value, all_attr_lab, all_obj_lab)


        logger.info('Finished.')



if __name__=="__main__":
    main()
