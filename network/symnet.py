from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, logging, torch, json
import os.path as osp
from collections import OrderedDict

from network.base_network import *
from utils import utils
from utils import config as cfg




class Network(BaseNetwork):
    root_logger = logging.getLogger("network %s"%__file__.split('/')[-1])


    def __init__(self, dataloader, args, feat_dim=None):
        super(Network, self).__init__(dataloader, args, feat_dim)
        
        self.dset = self.dataloader.dataset
        self.attr_embedder = utils.Embedder(args.wordvec, self.dset.attrs, args.data)
        self.emb_dim = self.attr_embedder.emb_dim      # dim of wordvec (attr or obj)

        self.pos_attr_id    = tf.placeholder(tf.int32, shape=[None])
        self.pos_obj_id     = tf.placeholder(tf.int32, shape=[None])
        self.pos_image_feat = tf.placeholder(tf.float32, shape=[None, self.feat_dim])

        self.neg_attr_id    = tf.placeholder(tf.int32, shape=[None])
        self.neg_obj_id     = tf.placeholder(tf.int32, shape=[None])
        self.neg_image_feat = tf.placeholder(tf.float32, shape=[None, self.feat_dim])

        self.test_attr_id   = tf.placeholder(tf.int32, shape=[None], name='test_attr_id')
        self.test_obj_id    = tf.placeholder(tf.int32, shape=[None], name='test_obj_id')

        self.pos_obj_prediction = tf.placeholder(tf.float32, shape=[None, self.num_obj])

        self.lr = tf.placeholder(tf.float32)


    def rep_embedder(self, img, is_training, name):
        """transform ResNet feature to image representation"""
        img = self.MLP(img, self.args.rep_dim, is_training, name, hidden_layers=[])
        return img



    def transformer(self, rep, v_attr, is_training, name):
        """CoN or DecoN in our paper"""
        
        with tf.variable_scope(name) as scope:
            in_dim = out_dim = self.args.rep_dim


            if not self.args.no_attention:
                attention = self.MLP(v_attr, in_dim, is_training,
                    name='fc_attention', hidden_layers=self.args.fc_att)
                attention = tf.sigmoid(attention)

                rep = attention*rep + rep      # short connection

            hidden = tf.concat([rep, v_attr], axis=1)  # 512+300


            output = self.MLP(hidden, out_dim, is_training, 
                name='fc_out', hidden_layers=self.args.fc_compress)
            

        return output

    
    def attr_classification(self, emb, is_training, name='classifier'):
        with tf.variable_scope(name) as scope:
            score_A = self.MLP(emb, self.num_attr, is_training, "attr", 
                hidden_layers=self.args.fc_cls)
            prob_A = tf.nn.softmax(score_A, 1)
            return score_A, prob_A


    def obj_classification(self, emb, is_training, name='classifier'):
        with tf.variable_scope(name) as scope:
            score_O = self.MLP(emb, self.num_obj, is_training, "obj", 
                hidden_layers=self.args.fc_cls)
            prob_O = tf.nn.softmax(score_O, 1)
            return score_O, prob_O
    



    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat, is_training, metric):
        """return attribute classification probability with our RMD"""
        # feat_plus, feat_minus:  shape=(bz, #attr, dim_emb)
        # d_plus: distance between feature before&after CoN
        # d_minus: distance between feature before&after DecoN
        
        d_plus = self.distance_metric(feat_plus, repeat_img_feat)
        d_minus = self.distance_metric(feat_minus, repeat_img_feat)
        d_plus = tf.reshape(d_plus, [-1, self.num_attr]) # bz, #attr
        d_minus = tf.reshape(d_minus, [-1, self.num_attr]) # bz, #attr


        if metric == 'softmax':
            p_plus  = tf.nn.softmax(-d_plus, 1) # (bz, #attr), smaller = better
            p_minus = tf.nn.softmax(d_minus, 1) # (bz, #attr), larger = better
            return p_plus, p_minus
        
        
        elif metric == 'rmd':
            d_plus_comp  = self.dset.comp_gamma['b']*d_plus
            d_minus_comp = self.dset.comp_gamma['a']*d_minus
            d_plus_attr  = self.dset.attr_gamma['b']*d_plus
            d_minus_attr = self.dset.attr_gamma['a']*d_minus

            p_comp = tf.nn.softmax(d_minus_comp - d_plus_comp, axis=1)
            p_attr = tf.nn.softmax(d_minus_attr - d_plus_attr, axis=1)

            return p_comp, p_attr




    def build_network(self, test_only=False):
        logger = self.logger('create_train_arch')

        total_losses = []

        batchsize = tf.shape(self.pos_image_feat)[0]
        
        pos_attr_emb = self.attr_embedder.get_embedding(self.pos_attr_id)
        neg_attr_emb = self.attr_embedder.get_embedding(self.neg_attr_id)

        pos_img = self.rep_embedder(self.pos_image_feat, 
            is_training=True, name='embedder') # (bz,dim)
        neg_img = self.rep_embedder(self.neg_image_feat, 
            is_training=True, name='embedder') # (bz,dim)

        # rA = remove positive attribute A
        # aA = add positive attribute A
        # rB = remove negative attribute B
        # aB = add negative attribute B
        pos_aA = self.transformer(pos_img, pos_attr_emb, True, name='CoN')
        pos_aB = self.transformer(pos_img, neg_attr_emb, True, name='CoN')
        pos_rA = self.transformer(pos_img, pos_attr_emb, True, name='DeCoN')
        pos_rB = self.transformer(pos_img, neg_attr_emb, True, name='DeCoN')

        attr_emb = self.attr_embedder.get_embedding(np.arange(self.num_attr)) 
        # (#attr, dim_emb), wordvec of all attributes
        tile_attr_emb = utils.tile_tensor(attr_emb, 0, batchsize) 
        # (bz*#attr, dim_emb)
        

        ########################## classification losses ######################
        # unnecessary to compute cls loss for neg_img

        if self.args.lambda_cls_attr > 0: 

            # original image
            _, prob_pos_A = self.attr_classification(pos_img, is_training=True)
            loss_cls_pos_a = self.cross_entropy(
                prob_pos_A, self.pos_attr_id, self.num_attr, 
                weight=self.attr_weight)

            # after removing pos_attr
            _, prob_pos_rA_A = self.attr_classification(pos_rA, is_training=True)
            loss_cls_pos_rA_a = self.cross_entropy(
                prob_pos_rA_A, self.pos_attr_id, self.num_attr, 
                target=0, weight=self.attr_weight)


            # rmd
            repeat_img_feat = utils.repeat_tensor(pos_img, 0, self.num_attr)  
            # (bz*#attr, dim_rep)
            feat_plus  = self.transformer(repeat_img_feat, tile_attr_emb, 
                is_training=True, name='CoN')
            feat_minus = self.transformer(repeat_img_feat, tile_attr_emb, 
                is_training=True, name='DeCoN')
            
            prob_RMD_plus, prob_RMD_minus = self.RMD_prob(feat_plus, feat_minus, 
                repeat_img_feat, is_training=True, metric=self.args.rmd_metric)

            loss_cls_rmd_plus  = self.cross_entropy(
                prob_RMD_plus, self.pos_attr_id, self.num_attr,
                weight=self.attr_weight)
            loss_cls_rmd_minus = self.cross_entropy(
                prob_RMD_minus, self.pos_attr_id, self.num_attr,
                weight=self.attr_weight)
            
            

            loss_cls_attr = self.args.lambda_cls_attr * sum([
                loss_cls_pos_a, loss_cls_pos_rA_a,
                loss_cls_rmd_plus, loss_cls_rmd_minus
            ])

            total_losses.append(loss_cls_attr)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_cls_attr', loss_cls_attr)
                    


        if self.args.lambda_cls_obj > 0:
            # original image
            _, prob_pos_O = self.obj_classification(pos_img, is_training=True)
            loss_cls_pos_o = self.cross_entropy(
                prob_pos_O, self.pos_obj_id, self.num_obj, 
                weight=self.obj_weight)

            # after removing pos_attr
            _, prob_pos_rA_O = self.obj_classification(pos_rA, is_training=True)
            loss_cls_pos_rA_o = self.cross_entropy(
                prob_pos_rA_O, self.pos_obj_id, self.num_obj, 
                weight=self.obj_weight)
            
            # after adding neg_attr
            _, prob_pos_aB_O = self.obj_classification(pos_aB, is_training=True)
            loss_cls_pos_aB_o = self.cross_entropy(
                prob_pos_aB_O, self.pos_obj_id, self.num_obj, 
                weight=self.obj_weight)


            loss_cls_obj = self.args.lambda_cls_obj * sum([
                loss_cls_pos_o, 
                loss_cls_pos_rA_o,
                loss_cls_pos_aB_o
            ])

            total_losses.append(loss_cls_obj)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_cls_obj', loss_cls_obj)
                    


        ############################# symmetry loss ###########################

        if self.args.lambda_sym > 0:

            loss_sym_pos = self.MSELoss(pos_aA, pos_img)
            loss_sym_neg = self.MSELoss(pos_rB, pos_img)

            loss_sym = self.args.lambda_sym * (loss_sym_pos + loss_sym_neg)
            total_losses.append(loss_sym)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_sym', loss_sym)


        ############################## axiom losses ###########################
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0
            
            # closure
            if not self.args.remove_clo:
                pos_aA_rA = self.transformer(pos_aA, pos_attr_emb, 
                    is_training=True, name='DeCoN')
                pos_rB_aB = self.transformer(pos_rB, neg_attr_emb, 
                    is_training=True, name='CoN')
                loss_clo = self.MSELoss(pos_aA_rA, pos_rA) + \
                            self.MSELoss(pos_rB_aB, pos_aB)

            # invertibility
            if not self.args.remove_inv:
                pos_rA_aA = self.transformer(pos_rA, pos_attr_emb, 
                    is_training=True, name='CoN')
                pos_aB_rB = self.transformer(pos_aB, neg_attr_emb, 
                    is_training=True, name='DeCoN')
                loss_inv = self.MSELoss(pos_rA_aA, pos_img) + \
                    self.MSELoss(pos_aB_rB, pos_img)

            # Commutativity
            if not self.args.remove_com:
                pos_aA_rB = self.transformer(pos_aA, neg_attr_emb, 
                    is_training=True, name='DeCoN')
                pos_rB_aA = self.transformer(pos_rB, pos_attr_emb, 
                    is_training=True, name='CoN')
                loss_com = self.MSELoss(pos_aA_rB, pos_rB_aA)


            loss_axiom = self.args.lambda_axiom * (
                loss_clo + loss_inv + loss_com)
            total_losses.append(loss_axiom)

            with tf.device("/cpu:0"):
                with tf.name_scope('loss'):
                    tf.summary.scalar('loss_axiom', loss_axiom)
                    tf.summary.scalar('loss_clo', loss_clo)
                    tf.summary.scalar('loss_inv', loss_inv)
                    tf.summary.scalar('loss_com', loss_com)
        

        ############################# triplet loss ############################
        
        if self.args.lambda_trip > 0:
            pos_triplet = tf.reduce_mean(
                self.triplet_margin_loss(pos_img, pos_aA, pos_rA))
            neg_triplet = tf.reduce_mean(
                self.triplet_margin_loss(pos_img, pos_rB, pos_aB))
            
            loss_triplet = self.args.lambda_trip * (pos_triplet + neg_triplet)
            total_losses.append(loss_triplet)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_triplet', loss_triplet)
        

        ################################ testing ##############################
        pos_img = self.rep_embedder(self.pos_image_feat, False, "embedder")
        repeat_img_feat = utils.repeat_tensor(pos_img, 0, self.num_attr)  
        # (bz*#attr, dim_rep)

        feat_plus = self.transformer(repeat_img_feat, tile_attr_emb, 
            is_training=False, name='CoN')
        feat_minus = self.transformer(repeat_img_feat, tile_attr_emb, 
            is_training=False, name='DeCoN')

        prob_A_rmd, prob_A_attr = self.RMD_prob(feat_plus, feat_minus, 
            repeat_img_feat, is_training=False, metric='rmd')

        _, prob_A_fc = self.attr_classification(pos_img, is_training=False)
        _, prob_O_fc = self.obj_classification(pos_img, is_training=False)


        

        if self.args.obj_pred is None:
            prob_O = prob_O_fc 
        else:
            prob_O = self.pos_obj_prediction
        
        test_a_onehot = tf.one_hot(self.test_attr_id, depth=self.num_attr, axis=1)
        test_o_onehot = tf.one_hot(self.test_obj_id, depth=self.num_obj, axis=1)
        # (n_p, n_o)
        prob_P_rmd = tf.multiply(
            tf.matmul(prob_A_rmd, tf.transpose(test_a_onehot)), 
            tf.matmul(prob_O, tf.transpose(test_o_onehot))
        )

        score_res = dict([
            ("score_rmd",   [prob_P_rmd, prob_A_attr, prob_O]),
        ])

        ############################### summary ###############################

        loss = sum(total_losses)
        
        with tf.device("/cpu:0"):
            tf.summary.scalar('loss_total', loss)
            tf.summary.scalar('lr', self.lr)
        

        train_summary_op = tf.summary.merge_all()

        
        return loss, score_res, train_summary_op




    def train_step(self, sess, blobs, lr, train_op, train_summary_op):
        
        summary, _ = sess.run(
            [train_summary_op, train_op],
            feed_dict={
                self.pos_attr_id     : blobs[1],
                self.pos_obj_id      : blobs[2],
                self.pos_image_feat  : blobs[4],
                self.neg_attr_id     : blobs[6],
                self.neg_image_feat  : blobs[9],
                self.lr: lr,
            })

        return summary

    
    
    
