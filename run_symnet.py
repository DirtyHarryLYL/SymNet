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



def make_parser():
    parser = argparse.ArgumentParser()

    # basic training settings
    parser.add_argument("--name", type=str, required=True, 
        help="Experiment name")
    parser.add_argument("--data", type=str, required=True,
        choices=['MIT','UT','MITg','UTg','APY','SUN'],
        help="Dataset name")
    parser.add_argument("--network", type=str, default='symnet', 
        help="Network name (the file name in `network` folder, but without suffix `.py`)")


    parser.add_argument("--epoch", type=int, default=500,
        help="Maximum epoch during training phase, or epoch to be tested during testing")
    parser.add_argument("--bz", type=int, default=512,
        help="Train batch size")
    parser.add_argument("--test_bz", type=int, default=1024,
        help="Test batch size")

    parser.add_argument("--trained_weight", type=str, default=None,
        help="Restore from a certain trained weight (relative path to './weights')")
    parser.add_argument("--weight_type", type=int, default=1, 
        help="Type of the trained weight: 1-previous checkpoint(default), 2-pretrained classifier, 3-pretrained transformer")

    parser.add_argument("--test_freq", type=int, default=1,
        help="Frequency of testing (#epoch). Set to 0 to skip test phase")
    parser.add_argument("--snapshot_freq", type=int, default=10,
        help="Frequency of saving snapshots (#epoch)")

    parser.add_argument("--force", default=False, action='store_true',
        help="WARINING: clear experiment with duplicated name")
    
    
    # model settings

    parser.add_argument("--rmd_metric", type=str, default='softmax',
        help="Similarity metric in RMD classification")
    parser.add_argument("--distance_metric", type=str, default='L2', 
        help="Distance form")

    parser.add_argument("--obj_pred", type=str, default=None, 
        help="Object prediction from pretrained model")

    parser.add_argument("--wordvec", type=str, default='glove',
        help="Pre-extracted word vector type")
    


    # important hyper-parameters
    
    parser.add_argument("--rep_dim", type=int, default=300,
        help="Dimentionality of attribute/object representation")
    
    parser.add_argument("--lr", type=float, default=1e-3,
        help="Learning rate")

    parser.add_argument("--dropout", type=float, 
        help="Keep probability of dropout")
    parser.add_argument("--batchnorm", default=False, action='store_true',
        help="Use batch normalization")
    parser.add_argument("--loss_class_weight", default=True, action='store_true', 
        help="Add weight between classes (default=true)")
        
    parser.add_argument("--fc_att", type=int, default=[512], nargs='*',
        help="#fc layers after word vector")
    parser.add_argument("--fc_compress", type=int, default=[768], nargs='*',
        help="#fc layers after hidden layer")
    parser.add_argument("--fc_cls", type=int, default=[512], nargs='*',
        help="#fc layers in classifiers")
    
    
    
    parser.add_argument("--lambda_cls_attr", type=float, default=0)
    parser.add_argument("--lambda_cls_obj", type=float, default=0)

    parser.add_argument("--lambda_trip", type=float, default=0)
    parser.add_argument("--triplet_margin", type=float, default=0.5,
        help="Triplet loss margin")

    parser.add_argument("--lambda_sym", type=float, default=0)

    parser.add_argument("--lambda_axiom", type=float, default=0)
    parser.add_argument("--remove_inv", default=False, action="store_true")
    parser.add_argument("--remove_com", default=False, action="store_true")
    parser.add_argument("--remove_clo", default=False, action="store_true")

    parser.add_argument("--no_attention", default=False, action="store_true")
    


    # not so important
    parser.add_argument("--activation", type=str, default='relu',
        help="Activation function (relu, elu, leaky_relu, relu6)")
    parser.add_argument("--initializer", type=float, default=None,
        help="Weight initializer for fc (default=xavier init, set a float number to use Gaussian init)")
    parser.add_argument("--optimizer", type=str, default='sgd', 
        choices=['sgd', 'adam', 'adamw', 'momentum', 'rmsprop'],
        help="Type of optimizer (sgd, adam, momentum)")

    parser.add_argument("--lr_decay_type", type=str, default='no',
        help="Type of Learning rate decay: no/exp/cos")
    parser.add_argument("--lr_decay_step", type=int, default=100,
        help="The first learning rate decay step (only for cos/exp)")
    parser.add_argument("--lr_decay_rate", type=float, default=0.9,
        help="Decay rate of cos/exp")

    parser.add_argument("--focal_loss", type=float,
        help="`gamma` in focal loss. default=0 (=CE)")
    
    parser.add_argument("--clip_grad", default=False, action='store_true',
        help="Use gradient clipping")

    
    return parser



def main():
    logger = logging.getLogger('MAIN')

    parser = make_parser()
    args = parser.parse_args()
    utils.duplication_check(args)
    utils.display_args(args, logger)


    logger.info("Loading dataset")


    train_dataloader = dataset.get_dataloader(args.data, 'train', 
        batchsize=args.bz)
    test_dataloader = dataset.get_dataloader(args.data, 'test', 
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(train_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, train_dataloader, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)




################################################################################



class SolverWrapper(BaseSolver):
    def __init__(self, network, train_dataloader, test_dataloader, args):
        logger = self.logger("init")
        self.network = network

        if args.network == 'fc_obj':
            self.criterion = 'real_obj_acc'
            self.key_score_name = "score_fc"
        else:
            self.criterion = 'top1_acc'
            self.key_score_name = "score_rmd"
        

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.weight_dir = osp.join(cfg.WEIGHT_ROOT_DIR, args.name)
        self.log_dir = osp.join(cfg.LOG_ROOT_DIR, args.name)

        self.args = args

        logger.info("training weight  => "+self.weight_dir)
        logger.info("training log     => "+self.log_dir)


        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if  args.trained_weight is not None:
            self.trained_weight = os.path.join(cfg.WEIGHT_ROOT_DIR, args.trained_weight)
            self.logger("init").info("pretrained model <= "+self.trained_weight)
            if args.weight_type != 1:
                self.clear_folder()
        else:
            self.trained_weight = None
            self.clear_folder()

        

    def construct_graph(self, sess):
        logger = self.logger('construct_graph')

        with sess.graph.as_default():
            if cfg.RANDOM_SEED is not None:
                tf.set_random_seed(cfg.RANDOM_SEED)

            loss_op, score_op, train_summary_op = self.network.build_network()
            

            self.epoch_num = 0
            if self.trained_weight is not None and self.args.weight_type == 1:
                try:
                    self.epoch_num = int(self.trained_weight.split('.ckpt')[0].split('_')[-1])
                except:
                    pass
            
            global_step    = tf.Variable(self.epoch_num, trainable=False)

            lr = self.set_lr_decay(global_step)
            optimizer = self.set_optimizer(lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies([tf.group(*update_ops)]):
                grad = optimizer.compute_gradients(loss_op, tf.trainable_variables())
                if self.args.clip_grad:
                    grad = [(tf.clip_by_norm(g, cfg.GRADIENT_CLIPPING), var) for g, var in grad]

                train_op = optimizer.apply_gradients(grad, global_step=global_step)


            self.saver = tf.train.Saver(max_to_keep=None)


        return lr, score_op, train_op, train_summary_op

        

    def trainval_model(self, sess, max_epoch):
        logger = self.logger('train_model')
        logger.info('Begin training')

        lr, score_op, train_op, train_summary_op = self.construct_graph(sess)
        #for x in tf.global_variables():
        #    print(x.name)

        self.initialize(sess)
        sess.graph.finalize()


        writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        evaluator = CZSL_Evaluator(self.test_dataloader.dataset, self.network)
        best_report = defaultdict(dict)


        for epoch in range(self.epoch_num+1, max_epoch+1):
            ############################# train ###############################
            summary = None
            for batch_ind, batch in tqdm.tqdm(
                    enumerate(self.train_dataloader), 
                    total=len(self.train_dataloader),
                    postfix='epoch %d/%d'%(epoch, max_epoch)):
                
                if isinstance(lr, float):
                    eval_lr = lr
                else:
                    eval_lr = lr.eval()
                
                summary = self.network.train_step(
                    sess, batch, eval_lr,
                    train_op, train_summary_op)

            writer.add_summary(summary, float(epoch))


            if self.args.test_freq>0 and epoch % self.args.test_freq == 0:
                ########################## test czsl ##########################

                accuracies_pair = defaultdict(list)
                accuracies_attr = defaultdict(list)
                accuracies_obj = defaultdict(list)

                for image_ind, batch in tqdm.tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader), postfix='test'):

                    predictions = self.network.test_step(sess, batch, score_op)

                    attr_truth = torch.from_numpy(batch[1])
                    obj_truth  = torch.from_numpy(batch[2])

                    for key in score_op.keys():
                        p_pair, p_a, p_o = predictions[key]
                        pair_results = evaluator.score_model(p_pair, obj_truth)
                        match_stats = evaluator.evaluate_predictions(
                            pair_results, attr_truth, obj_truth)
                        accuracies_pair[key].append(match_stats)
                        # 0/1 sequence of t/f

                        a_match, o_match = evaluator.evaluate_only_attr_obj(
                            p_a, attr_truth, p_o, obj_truth)

                        accuracies_attr[key].append(a_match)
                        accuracies_obj[key].append(o_match)

                        

                all_report = {'Current':{}, 'Best':{}}

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
                        'epoch':        epoch,
                    }

                    if self.criterion not in best_report[name] or report_dict[self.criterion] > best_report[name][self.criterion]:
                        best_report[name] = report_dict

                    # print test results
                    all_report['Current'][name] = report_dict
                    all_report['Best'][name] = best_report[name]
                    for key in all_report.keys():
                        print("%s %s: "%(key, name) + utils.formated_czsl_result(all_report[key][name]))

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
