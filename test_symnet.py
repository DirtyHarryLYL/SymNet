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
from utils.evaluator import CZSL_Evaluator


from run_symnet import make_parser


def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    utils.display_args(args, logger)


    logger.info("Loading dataset")
    test_dataloader = dataset.get_dataloader(args.data, 'test',
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(test_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)




################################################################################



class SolverWrapper(BaseSolver):
    def __init__(self, network, test_dataloader, args):
        logger = self.logger("init")
        self.network = network

        self.test_dataloader = test_dataloader
        self.weight_dir = osp.join(cfg.WEIGHT_ROOT_DIR, args.name)
        self.args = args


        self.trained_weight = os.path.join(cfg.WEIGHT_ROOT_DIR, args.name, "snapshot_epoch_%d.ckpt"%args.epoch)
        self.logger("init").info("pretrained model <= "+self.trained_weight)
            

        

    def construct_graph(self, sess):
        logger = self.logger('construct_graph')

        with sess.graph.as_default():
            if cfg.RANDOM_SEED is not None:
                tf.set_random_seed(cfg.RANDOM_SEED)

            loss_op, score_op, train_summary_op = self.network.build_network()
            
            global_step = tf.Variable(self.args.epoch, trainable=False)

        return score_op, train_summary_op

        

    def trainval_model(self, sess, max_epoch):
        logger = self.logger('train_model')
        logger.info('Begin training')

        score_op, train_summary_op = self.construct_graph(sess)
        #for x in tf.global_variables():
        #    print(x.name)

        self.initialize(sess)
        sess.graph.finalize()

        evaluator = CZSL_Evaluator(self.test_dataloader.dataset, self.network)


        ############################## test czsl ################################

        accuracies_pair = defaultdict(list)
        accuracies_attr = defaultdict(list)
        accuracies_obj = defaultdict(list)

        for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

            predictions = self.network.test_step(sess, batch, score_op)  # ordereddict of [score_pair, score_a, score_o]

            attr_truth, obj_truth = batch[1], batch[2]
            attr_truth, obj_truth = torch.from_numpy(attr_truth), torch.from_numpy(obj_truth)

            match_stats = []
            for key in score_op.keys():
                p_pair, p_a, p_o = predictions[key]
                pair_results = evaluator.score_model(p_pair, obj_truth)
                match_stats = evaluator.evaluate_predictions(pair_results, attr_truth, obj_truth)
                accuracies_pair[key].append(match_stats)  # 0/1 sequence of t/f

                a_match, o_match = evaluator.evaluate_only_attr_obj(p_a, attr_truth, p_o, obj_truth)

                accuracies_attr[key].append(a_match)
                accuracies_obj[key].append(o_match)



        for name in accuracies_pair.keys():
            accuracies = accuracies_pair[name]
            accuracies = zip(*accuracies)
            accuracies = map(torch.mean, map(torch.cat, accuracies))
            attr_acc, obj_acc, closed_1_acc, closed_2_acc, closed_3_acc, _, objoracle_acc = map(lambda x:x.item(), accuracies)

            real_attr_acc = torch.mean(torch.cat(accuracies_attr[name])).item()
            real_obj_acc = torch.mean(torch.cat(accuracies_obj[name])).item()

            report_dict = {
                'real_attr_acc':real_attr_acc,
                'real_obj_acc': real_obj_acc,
                'top1_acc':     closed_1_acc,
                'top2_acc':     closed_2_acc,
                'top3_acc':     closed_3_acc,
                'name':         self.args.name,
                'epoch':        self.args.epoch,
            }

            print(name + ": " + utils.formated_czsl_result(report_dict))
                    

        logger.info('Finished.')






if __name__=="__main__":
    main()
