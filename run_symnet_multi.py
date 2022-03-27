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

from run_symnet import make_parser as make_basic_parser
from run_symnet import SolverWrapper as CZSLSolverWrapper




def make_parser():
    parser = make_basic_parser()
    parser.add_argument("--lambda_atten", type=float, default=0)
    parser.add_argument("--lambda_multi_rmd", type=float, default=0)
    parser.add_argument("--sample_bar", default=0.1, type=float)
    parser.add_argument("--bce_neg_weight", default=0., type=float)
    return parser


def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    args.network = 'symnet_multi'
    
    utils.duplication_check(args)
    utils.display_args(args, logger)
    logger.info("Loading dataset")
    

    train_dataloader = dataset.get_dataloader(args.data, 'train', 
        batchsize=args.bz,args=args)
    test_dataloader = dataset.get_dataloader(args.data, 'test', 
        batchsize=args.test_bz, args=args)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(train_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, train_dataloader, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)


class SolverWrapper(CZSLSolverWrapper):
        

    def trainval_model(self, sess, max_epoch):
        self.criterion = 'mAUC'
        logger = self.logger('train_model')
        logger.info('Begin training')

        lr, score_op, train_op, train_summary_op = self.construct_graph(sess)
        self.initialize(sess)
        sess.graph.finalize()


        writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        evaluator = Multi_Evaluator()

        best_report = defaultdict(dict)

        for epoch in range(self.epoch_num+1, max_epoch+1):
            ################################## train ####################################
            summary = None
            for batch_ind, batch in tqdm.tqdm(enumerate(self.train_dataloader), 
                                            total=len(self.train_dataloader), 
                                            postfix='epoch %d/%d'%(epoch, max_epoch)):
                
                if isinstance(lr, float):
                    eval_lr = lr
                else:
                    eval_lr = lr.eval()
                
                summary = self.network.train_step(sess, batch, eval_lr, train_op, train_summary_op)

            writer.add_summary(summary, float(epoch))


            if self.args.test_freq>0 and epoch % self.args.test_freq == 0:
                ############################## test ################################
                
                all_attr = []
                all_pred = defaultdict(list)

                for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

                    predictions = self.network.test_step(sess, batch, score_op) 

                    all_attr.append(torch.from_numpy(batch[1]))
                    for key in score_op.keys():
                        all_pred[key].append(predictions[key])

                
                all_report = {'Current':{}, 'Best':{}}
                for name, pred in all_pred.items():
                    mAP, mAUC = evaluator(
                        np.concatenate(pred,0),
                        np.concatenate(all_attr,0)
                    )
                    # save to tensorboard
                    report_dict = {
                        'mAP':          mAP,
                        'mAUC':         mAUC,
                        'name':         self.args.name,
                        'epoch':        epoch,
                    }

                    if self.criterion not in best_report[name] or report_dict[self.criterion] > best_report[name][self.criterion]:
                        best_report[name] = report_dict
                    
                    # print test results
                    all_report['Current'][name] = report_dict
                    all_report['Best'][name] = best_report[name]
                    for key in all_report.keys():
                        print("%s %s: "%(key, name) + utils.formated_multi_result(all_report[key][name]))

                    # save to tensorboard
                    summary = tf.Summary()
                    for key, value in report_dict.items():
                        if key not in ['name', 'epoch']:
                            summary.value.add(tag="%s/%s"%(name,key), simple_value=value)
                    writer.add_summary(summary, epoch)

            if epoch % self.args.snapshot_freq == 0:
                self.snapshot(sess, epoch)


        writer.close()
        logger.info('Finished.')






if __name__=="__main__":
    main()