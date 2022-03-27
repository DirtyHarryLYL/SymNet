from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, logging, torch, json
import os.path as osp
from collections import OrderedDict
from tensorflow.python.ops import array_ops
from utils import utils
from utils import config as cfg
from network.symnet import Network as BaseNetwork
import tensorflow as tf


class Network(BaseNetwork):
    root_logger = logging.getLogger("network %s" % __file__.split('/')[-1])

    def __init__(self, dataloader, args, feat_dim=None):
        super(Network, self).__init__(dataloader, args, feat_dim)
        self.gt_attr_vec = tf.placeholder(tf.float32, shape=[None, self.num_attr], name='gt_attr')
        self.gt_obj_id = tf.placeholder(tf.int32, shape=[None], name='gt_obj')
        self.attr_a = tf.placeholder(tf.int32, shape=[None], name='attr_a')
        self.attr_b = tf.placeholder(tf.int32, shape=[None], name='attr_b')
        self.attr_c = tf.placeholder(tf.int32, shape=[None], name='attr_c')
        self.img_a_sim = tf.placeholder(tf.float32, shape=[None, self.num_attr], name='img_a_sim')

        self.attr_pair_a = self.dataloader.dataset.attr_pair[:, 0]
        self.attr_pair_b = self.dataloader.dataset.attr_pair[:, 1]

        self.num_pair = self.attr_pair_a.shape[0]
        self.part_pair_mask = tf.placeholder(tf.float32, shape=[None, self.num_pair], name='part_pair_mask')
        self.att_sim = self.dataloader.dataset.att_sim_matrix

    def RMD_prob(self, feat_plus, feat_minus, repeat_img_feat, is_training, metric='rmd'):
        """return attribute classification probability with our RMD"""

        def diff_log(d_plus, d_minus):
            if not is_training:
                return
            with tf.device("/cpu:0"):
                with tf.name_scope('delta_dist'):
                    tf.summary.histogram('plus', d_plus)
                    tf.summary.histogram('minus', d_minus)
                    tf.summary.histogram('delta', d_minus - d_plus)


        d_plus = self.distance_metric(feat_plus, repeat_img_feat)
        d_minus = self.distance_metric(feat_minus, repeat_img_feat)
        d_plus = tf.reshape(d_plus, [-1, self.num_attr])  # bz, #attr
        d_minus = tf.reshape(d_minus, [-1, self.num_attr])  # bz, #attr

        # if not is_training and not self.args.no_fuse:
        #     d_plus = tf.matmul(d_plus, self.dset.trained_corre_fuse['fuse_dp'])
        #     d_minus = tf.matmul(d_minus, self.dset.trained_corre_fuse['fuse_dm'])

        if metric == 'raw_distance':
            return d_plus, d_minus
        elif metric == 'rmd':
            if not is_training:
                d_minus = d_minus * self.dset.gamma['attr_a']
                d_plus = d_plus * self.dset.gamma['attr_b']
            diff = d_minus - d_plus
            p_diff = tf.sigmoid(diff)
            return p_diff, p_diff

        elif metric == 'sigmoid':
            return tf.sigmoid(-d_plus), tf.sigmoid(d_minus)

        else:
            raise NotImplementedError('metric %s is not implemented yet' % metric)

    def attr_classification(self, emb, is_training, name='classifier'):
        with tf.variable_scope(name) as scope:
            score_A = self.MLP(emb, self.num_attr, is_training, "attr",
                               hidden_layers=self.args.fc_cls)
            prob_A = tf.sigmoid(score_A)
            return score_A, prob_A

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
            
        return output, attention

    def build_network(self):
        total_losses = []

        batchsize = tf.shape(self.pos_image_feat)[0]

        img_feat = self.rep_embedder(self.pos_image_feat, True, "embedder")
        repeat_img_feat = utils.repeat_tensor(img_feat, 0, self.num_attr)  # (bz*#attr, dim_rep)

        gt_attr_vec = self.gt_attr_vec
        gt_obj_vec = tf.one_hot(self.gt_obj_id, depth=self.num_obj, axis=1)

        attr_emb = self.attr_embedder.get_embedding(
            np.arange(self.num_attr))  # (#attr, dim_emb), wordvec of all attributes
        tile_attr_emb = utils.tile_tensor(attr_emb, 0, batchsize)  # (bz*#attr, dim_emb)

        feat_plus, tile_atten_plus = self.transformer(repeat_img_feat, tile_attr_emb, is_training=True, name='CoN')
        feat_minus, tile_atten_minus = self.transformer(repeat_img_feat, tile_attr_emb, is_training=True, name='DeCoN')
        tile_atten_minus = tf.reshape(tile_atten_minus, (batchsize, self.num_attr, self.args.rep_dim))
        tile_atten_plus = tf.reshape(tile_atten_plus, (batchsize, self.num_attr, self.args.rep_dim))

        attr_sim_diff = tf.gather_nd(self.att_sim, indices=tf.stack([self.attr_a, self.attr_b], axis=1)) - \
                        tf.gather_nd(self.att_sim, indices=tf.stack([self.attr_a, self.attr_c], axis=1))

        feat_plus_minus, _ = self.transformer(feat_plus, tile_attr_emb,
                                                 is_training=True,
                                                 name='DeCoN')
        feat_minus_plus, _ = self.transformer(feat_minus, tile_attr_emb,
                                                 is_training=True,
                                                 name='CoN')

        ########################## classification losses ######################
        if self.args.lambda_cls_attr > 0:
            repeat_gt_attr_vec = utils.repeat_tensor(gt_attr_vec, 0, self.num_attr)  # (bz*#attr, #attr)
            transform_attr_onehot = utils.tile_tensor(tf.eye(self.num_attr), 0, batchsize)
            ###### origin
            _, prob_orig_A = self.attr_classification(img_feat, is_training=True, name='classifier')
            loss_cls_orig_a = self.cross_entropy_with_labelvec(prob_orig_A, gt_attr_vec, weight=self.attr_weight,
                                                                neg_loss=self.args.bce_neg_weight)
        

            ###### plus
            attr_labelvec_plus = tf.minimum(1., repeat_gt_attr_vec + transform_attr_onehot)
            _, prob_plus_A = self.attr_classification(feat_plus, is_training=True, name='classifier')
            loss_cls_plus_a = self.cross_entropy_with_labelvec(prob_plus_A, attr_labelvec_plus,
                                                               weight=self.attr_weight,
                                                               neg_loss=self.args.bce_neg_weight,
                                                               mask=transform_attr_onehot)

            loss_cls_plus_a_carlib = self.carlibration_look_up(prob_plus_A, transform_attr_onehot, 1)
            loss_cls_plus_a = loss_cls_plus_a + loss_cls_plus_a_carlib

            ###### minus
            attr_labelvec_minus = tf.maximum(0., repeat_gt_attr_vec - transform_attr_onehot)
            _, prob_minus_A = self.attr_classification(feat_minus, is_training=True, name='classifier')
            loss_cls_minus_a = self.cross_entropy_with_labelvec(prob_minus_A, attr_labelvec_minus,
                                                                weight=self.attr_weight,
                                                                neg_loss=self.args.bce_neg_weight,
                                                                mask=transform_attr_onehot)

            loss_cls_minus_a_carlib = self.carlibration_look_up(prob_minus_A, transform_attr_onehot, 0)
            loss_cls_minus_a = loss_cls_minus_a + loss_cls_minus_a_carlib

            ###### rmd
            prob_RMD_plus, prob_RMD_minus = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat,
                                                          is_training=True, metric=self.args.rmd_metric)[:2]
            prob_RMD = (prob_RMD_minus + prob_RMD_plus) / 2.
            loss_cls_rmd = self.cross_entropy_with_labelvec(prob_RMD, gt_attr_vec, weight=self.attr_weight,
                                                            neg_loss=self.args.bce_neg_weight)

            loss_cls_attr = self.args.lambda_cls_attr * sum([
                loss_cls_orig_a, loss_cls_plus_a,
                loss_cls_minus_a, loss_cls_rmd
            ])

            total_losses.append(loss_cls_attr)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_cls_attr', loss_cls_attr)
                tf.summary.scalar('loss_cls_attr/loss_cls_orig_a', loss_cls_orig_a)
                tf.summary.scalar('loss_cls_attr/loss_cls_plus_a', loss_cls_plus_a)
                tf.summary.scalar('loss_cls_attr/loss_cls_minus_a', loss_cls_minus_a)
                tf.summary.scalar('loss_cls_attr/loss_cls_rmd', loss_cls_rmd)

        if self.args.lambda_cls_obj > 0:
            repeat_gt_obj_vec = utils.repeat_tensor(gt_obj_vec, 0, self.num_attr)  # (bz*#attr, #obj)
            ###### origin
            _, prob_orig_O = self.obj_classification(img_feat, is_training=True, name='classifier')
            loss_cls_orig_o = self.cross_entropy_with_labelvec(prob_orig_O, gt_obj_vec, weight=self.obj_weight)
            ###### plus
            _, prob_plus_O = self.obj_classification(feat_plus, is_training=True,
                                                     name='classifier')  # (bz*#attr, #attr)
            loss_cls_plus_o = self.cross_entropy_with_labelvec(prob_plus_O, repeat_gt_obj_vec, weight=self.obj_weight)
            ###### minus
            _, prob_minus_O = self.obj_classification(feat_minus, is_training=True,
                                                      name='classifier')  # (bz*#attr, #attr)
            loss_cls_minus_o = self.cross_entropy_with_labelvec(prob_minus_O, repeat_gt_obj_vec, weight=self.obj_weight)

            loss_cls_obj = self.args.lambda_cls_obj * sum([
                loss_cls_orig_o, loss_cls_plus_o, loss_cls_minus_o
            ])
            total_losses.append(loss_cls_obj)
            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_cls_obj', loss_cls_obj)

        ################################################ symmetry losses ###############################################
        if self.args.lambda_sym > 0:
            dist_sym_plus = tf.reshape(self.distance_metric(repeat_img_feat, feat_plus), [batchsize, self.num_attr])
            dist_sym_minus = tf.reshape(self.distance_metric(repeat_img_feat, feat_minus), [batchsize, self.num_attr])
            # both have shape (bz, #attr)

            dist_sym_plus = dist_sym_plus * gt_attr_vec
            dist_sym_minus = dist_sym_minus * (1 - gt_attr_vec)

            loss_sym_pos = tf.reduce_mean(dist_sym_plus, axis=0)
            loss_sym_neg = tf.reduce_mean(dist_sym_minus, axis=0)

            loss_sym_pos = tf.reduce_sum(loss_sym_pos)
            loss_sym_neg = tf.reduce_sum(loss_sym_neg)

            loss_sym = self.args.lambda_sym * (loss_sym_pos + loss_sym_neg)
            total_losses.append(loss_sym)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_sym', loss_sym)

        #################################################  axiom losses ################################################
        if self.args.lambda_axiom > 0:
            loss_clo = loss_inv = loss_com = 0
            ################################################# inv loss #################################################
            if not self.args.remove_inv:
                dist_plus_minus = tf.reshape(self.distance_metric(repeat_img_feat, feat_plus_minus),
                                             [batchsize, self.num_attr])
                dist_minus_plus = tf.reshape(self.distance_metric(repeat_img_feat, feat_minus_plus),
                                             [batchsize, self.num_attr])

                dist_inv_plus = dist_minus_plus * gt_attr_vec
                dist_inv_minus = dist_plus_minus * (1 - gt_attr_vec)

                loss_inv_pos = tf.reduce_mean(dist_inv_plus, axis=0)
                loss_inv_neg = tf.reduce_mean(dist_inv_minus, axis=0)

                loss_inv_pos = tf.reduce_sum(loss_inv_pos)
                loss_inv_neg = tf.reduce_sum(loss_inv_neg)

                loss_inv = loss_inv_pos + loss_inv_neg

            ############################################### closure loss ###############################################
            if not self.args.remove_clo:
                dist_clo_plus = tf.reshape(self.distance_metric(feat_minus, feat_plus_minus),
                                           [batchsize, self.num_attr])
                dist_clo_minus = tf.reshape(self.distance_metric(feat_plus, feat_minus_plus),
                                            [batchsize, self.num_attr])
                loss_clo_pos = tf.reduce_mean(dist_clo_plus * gt_attr_vec, axis=0)
                loss_clo_neg = tf.reduce_mean(dist_clo_minus * (1 - gt_attr_vec), axis=0)

                loss_clo_pos = tf.reduce_sum(loss_clo_pos)
                loss_clo_neg = tf.reduce_sum(loss_clo_neg)

                loss_clo = loss_clo_pos + loss_clo_neg

            loss_axiom = (loss_clo + loss_inv) * self.args.lambda_axiom
            total_losses.append(loss_axiom)
            with tf.device("/cpu:0"):
                with tf.name_scope('loss'):
                    tf.summary.scalar('loss_axiom', loss_axiom)
                with tf.name_scope('loss_axiom'):
                    tf.summary.scalar('loss_clo', loss_clo)
                    tf.summary.scalar('loss_inv', loss_inv)

        ################################################  triplet losses ###############################################
        if self.args.lambda_trip > 0:
            triplet_pos = tf.reshape(self.triplet_margin_loss(repeat_img_feat, feat_plus, feat_minus),
                                     [batchsize, self.num_attr])
            triplet_neg = tf.reshape(self.triplet_margin_loss(repeat_img_feat, feat_minus, feat_plus),
                                     [batchsize, self.num_attr])

            loss_trip_pos = tf.reduce_mean(triplet_pos * gt_attr_vec, axis=0)
            loss_trip_neg = tf.reduce_mean(triplet_neg * (1 - gt_attr_vec), axis=0)

            loss_trip_pos = tf.reduce_sum(loss_trip_pos)
            loss_trip_neg = tf.reduce_sum(loss_trip_neg)

            loss_trip = self.args.lambda_trip * (loss_trip_pos + loss_trip_neg)
            total_losses.append(loss_trip)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_triplet', loss_trip)

        if self.args.lambda_multi_rmd > 0:
            dist_minus = tf.reshape(self.distance_metric(repeat_img_feat, feat_minus), [batchsize, self.num_attr])

            attr_pair_a = tf.tile(tf.expand_dims(self.attr_pair_a, 0), (batchsize, 1))
            attr_pair_b = tf.tile(tf.expand_dims(self.attr_pair_b, 0), (batchsize, 1))

            # bz * num_pair
            idx_a = tf.stack([tf.stack([tf.range(batchsize)] * self.num_pair, axis=1), attr_pair_a], axis=2)
            idx_b = tf.stack([tf.stack([tf.range(batchsize)] * self.num_pair, axis=1), attr_pair_b], axis=2)

            dist_a = tf.gather_nd(dist_minus, idx_a)
            dist_b = tf.gather_nd(dist_minus, idx_b)

            sim_a = tf.gather_nd(self.img_a_sim, idx_a)
            sim_b = tf.gather_nd(self.img_a_sim, idx_b)

            loss_multi_rmd = self.triplet_margin_loss_with_distance(dist_a, dist_b, weight=sim_b - sim_a,
                                                                    margin=self.args.triplet_margin)
            loss_multi_rmd = self.args.lambda_trip * self.args.lambda_multi_rmd * tf.reduce_mean(
                tf.reduce_sum(loss_multi_rmd * self.part_pair_mask, axis=1))

            total_losses.append(loss_multi_rmd)
            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_multi_rmd', loss_multi_rmd)

        ############################################# attention triplet losses #########################################
        if self.args.lambda_atten > 0:
            atten_a_p = tf.gather_nd(tile_atten_plus, tf.stack((tf.range(batchsize), self.attr_a), axis=1))
            atten_a_m = tf.gather_nd(tile_atten_minus, tf.stack((tf.range(batchsize), self.attr_a), axis=1))

            atten_b_p = tf.gather_nd(tile_atten_plus, tf.stack((tf.range(batchsize), self.attr_b), axis=1))
            atten_c_p = tf.gather_nd(tile_atten_plus, tf.stack((tf.range(batchsize), self.attr_c), axis=1))

            atten_b_m = tf.gather_nd(tile_atten_minus, tf.stack((tf.range(batchsize), self.attr_b), axis=1))
            atten_c_m = tf.gather_nd(tile_atten_minus, tf.stack((tf.range(batchsize), self.attr_c), axis=1))

            con_atten_triplet = tf.reduce_mean(
                self.triplet_margin_loss(atten_a_p, atten_b_p, atten_c_p,
                                         weight=tf.cast(attr_sim_diff, dtype=tf.float32)))
            decon_atten_triplet = tf.reduce_mean(
                self.triplet_margin_loss(atten_a_m, atten_b_m, atten_c_m,
                                         weight=tf.cast(attr_sim_diff, dtype=tf.float32)))

            loss_attention_trip = self.args.lambda_trip * self.args.lambda_atten * (con_atten_triplet + decon_atten_triplet)
            total_losses.append(loss_attention_trip)

            with tf.device("/cpu:0"):
                tf.summary.scalar('loss/loss_attention_trip', loss_attention_trip)

        loss = sum(total_losses)

        ####################################################### test ###################################################

        img_feat = self.rep_embedder(self.pos_image_feat, False, "embedder")
        repeat_img_feat = utils.repeat_tensor(img_feat, 0, self.num_attr)  # (bz*#attr, dim_rep)

        feat_plus, _ = self.transformer(repeat_img_feat, tile_attr_emb, is_training=False, name='CoN')
        feat_minus, _ = self.transformer(repeat_img_feat, tile_attr_emb, is_training=False, name='DeCoN')

        prob_A_rmd_plus, prob_A_rmd_minus = self.RMD_prob(feat_plus, feat_minus, repeat_img_feat,
                                                          is_training=False)[:2]
        prob_A_rmd = (prob_A_rmd_plus + prob_A_rmd_minus) / 2.

        _, prob_fc = self.attr_classification(img_feat, is_training=False)
        score_res = OrderedDict([
            ("attr_rmd", prob_A_rmd),
            ("FC", prob_fc)
        ])

        with tf.device("/cpu:0"):
            tf.summary.scalar('loss_total', loss)
            tf.summary.scalar('lr', self.lr)

        train_summary_op = tf.summary.merge_all()

        return loss, score_res, train_summary_op

    def train_step(self, sess, blobs, lr, train_op, train_summary_op):

        summary, _ = sess.run(
            [train_summary_op, train_op],
            feed_dict={
                self.pos_image_feat: blobs[0],
                self.gt_attr_vec: blobs[1],
                self.gt_obj_id: blobs[2],
                self.img_a_sim: blobs[3],
                self.attr_a: blobs[4][:, 0],
                self.attr_b: blobs[4][:, 1],
                self.attr_c: blobs[4][:, 2],
                self.part_pair_mask: blobs[5],
                self.lr: lr,
            })

        return summary

    def test_step(self, sess, blobs, score_op):
        score = sess.run(score_op, feed_dict={
            self.pos_image_feat: blobs[0]
        })
        return score
