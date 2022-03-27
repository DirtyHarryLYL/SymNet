from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from utils.base_solver import BaseSolver

import os, logging, importlib, re, copy, random, tqdm, argparse
import os.path as osp
import cPickle as pickle
import numpy as np
from collections import defaultdict

import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

from utils import config as cfg
from utils import dataset, utils
from utils.evaluator import Multi_Evaluator

from run_symnet_multi import make_parser
from test_symnet import SolverWrapper as BaseSolverWrapper




def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    args.network = 'symnet_multi'
    utils.display_args(args, logger)
    logger.info("Loading dataset")
    
    test_dataloader = dataset.get_dataloader(args.data, 'test', 
        batchsize=args.test_bz, args=args)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(test_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)


class SolverWrapper(BaseSolverWrapper):
            


    def trainval_model(self, sess, max_epoch):
        logger = self.logger('test_model')
        logger.info('Begin testing')

        score_op, train_summary_op = self.construct_graph(sess)
        self.initialize(sess)
        sess.graph.finalize()


        evaluator = Multi_Evaluator()
        all_attr = []
        all_pred = defaultdict(list)

        for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

            predictions = self.network.test_step(sess, batch, score_op) 
            all_attr.append(torch.from_numpy(batch[1]))
            for key in score_op.keys():
                all_pred[key].append(predictions[key])

        
        for name, pred in all_pred.items():
            mAP, mAUC = evaluator(
                np.concatenate(pred,0),
                np.concatenate(all_attr,0)
            )
            report_dict = {
                'mAP':          mAP,
                'mAUC':         mAUC,
                'name':         self.args.name,
                'epoch':        self.args.epoch,
            }

            print("%s: "%(name) + utils.formated_multi_result(report_dict))
        logger.info('Finished.')






if __name__=="__main__":
    main()