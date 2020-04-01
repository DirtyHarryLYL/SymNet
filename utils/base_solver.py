from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, logging
from pprint import pprint

from utils import config as cfg

if cfg.ROOT_DIR.startswith('/home'):
    import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set tensorflow logger to WARNING level
import tensorflow as tf
import torch




class BaseSolver(object):
    root_logger = logging.getLogger('solver')
    def logger(self, suffix):
        return self.root_logger.getChild(suffix)
    

    def clear_folder(self):
        """clear weight and log dir"""
        logger = self.logger('clear_folder')

        for f in os.listdir(self.log_dir):
            logger.warning('Deleted log file ' + f)
            os.remove(os.path.join(self.log_dir, f))

        for f in os.listdir(self.weight_dir):
            logger.warning('Deleted weight file ' + f)
            os.remove(os.path.join(self.weight_dir, f))


    def snapshot(self, sess, iter, filenames = None):
        """save checkpoint"""
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)

        if filenames is None:
            filename = 'snapshot_epoch_{}.ckpt'.format(iter)
        else:
            filename = filenames
        pth = os.path.join(self.weight_dir, filename)
        self.saver.save(sess, pth)

        self.logger('snapshot').info('Wrote snapshot to: {}'.format(filename))


    def initialize(self, sess):
        """weight initialization"""
        logger = self.logger('initialize')

        if self.trained_weight is None:
            sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            logger.info('Restoring whole model snapshots from {:s}'.format(self.trained_weight))
            saver_restore = tf.train.Saver()
            saver_restore.restore(sess, self.trained_weight)

            
    

    def set_lr_decay(self, global_step):
        if self.args.lr_decay_type == 'no':
            lr = self.args.lr
        elif self.args.lr_decay_type == 'exp':
            decay_stepsize = len(self.train_dataloader)*self.args.lr_decay_step
            lr = tf.train.exponential_decay(
                self.args.lr,
                global_step, 
                decay_stepsize,
                self.args.lr_decay_rate, 
                staircase=True)
        elif self.args.lr_decay_type == 'cos':
            decay_stepsize = len(self.train_dataloader)*self.args.lr_decay_step
            lr = tf.train.cosine_decay_restarts(
                self.args.lr,
                global_step,
                decay_stepsize,
                t_mul=2.0,
                m_mul=0.8,
                alpha=0.1
            )
        
        return lr
    


    def set_optimizer(self, lr):
        if self.args.optimizer == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif self.args.optimizer == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, 0.9)
            logger.info('Using momentum optimizer')
        elif self.args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
            logger.info('Using Adam optimizer')
        elif self.args.optimizer == 'adamw':
            optimizer = tf.contrib.opt.AdamWOptimizer(5e-5, lr)
            logger.info('Using AdamW optimizer')
        elif self.args.optimizer == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(lr)
            logger.info('Using RMSProp optimizer')
        
        return optimizer
