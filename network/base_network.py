import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import array_ops

import numpy as np
import os, logging, torch



class BaseNetwork(object):
    def logger(self, suffix):
        return self.root_logger.getChild(suffix)

    def __init__(self, dataloader, args, feat_dim=None):
        self.dataloader = dataloader
        self.num_attr = len(dataloader.dataset.attrs)
        self.num_obj = len(dataloader.dataset.objs)
        if feat_dim is not None:
            self.feat_dim = feat_dim
        else:
            self.feat_dim = dataloader.dataset.feat_dim

        self.args = args
        self.dropout = args.dropout
        

        if self.args.loss_class_weight and self.args.data not in ['SUN','APY']:
            from utils.aux_data import load_loss_weight
            self.num_pair = len(dataloader.dataset.pairs)
            
            attr_weight, obj_weight, pair_weight = load_loss_weight(self.args.data)
            self.attr_weight = np.array(attr_weight, dtype=np.float32)
            self.obj_weight  = np.array(obj_weight, dtype=np.float32)
            self.pair_weight = np.array(pair_weight, dtype=np.float32)

        else:
            self.attr_weight = None
            self.obj_weight  = None
            self.pair_weight = None
        

    def basic_argscope(self, is_training):
        activation_list = {
            'relu': tf.nn.relu,
            'elu': tf.nn.elu,
        }
        if hasattr(tf.nn, "leaky_relu"):
            activation_list['leaky_relu'] = tf.nn.leaky_relu
        if hasattr(tf.nn, "relu6"):
            activation_list['relu6'] = tf.nn.relu6

        if self.args.initializer is None:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.random_normal_initializer(0, self.args.initializer)

        return slim.arg_scope(
            [slim.fully_connected],
            activation_fn = activation_list[self.args.activation],
            normalizer_fn = slim.batch_norm if self.args.batchnorm else None,
            normalizer_params={
                'is_training': is_training, 
                'decay': 0.95, 
                'fused':False
            },
        )


    def MLP(self, input_feat, output_dim, is_training, name, hidden_layers=[]):
        """multi-layer perceptron, 1 layers as default"""

        with self.basic_argscope(is_training):
            with tf.variable_scope(name) as scope:
                for i, size in enumerate(hidden_layers):
                    input_feat = slim.fully_connected(input_feat, size,
                        trainable=is_training, reuse=tf.AUTO_REUSE, scope="fc_%d"%i)
                        
                    if self.dropout is not None:
                        input_feat = slim.dropout(input_feat, keep_prob=self.dropout, is_training=is_training, scope='dropout_%d'%i)

                output_feat = slim.fully_connected(input_feat, output_dim,
                    trainable=is_training, activation_fn=None, reuse=tf.AUTO_REUSE, scope="fc_out")
            
        return output_feat
        

    def cross_entropy(self, prob, labels, depth, target=1, weight=None, sample_weight=None, neg_loss=0):
        """cross entropy with GT label id (int)"""
        onehot_label = tf.one_hot(labels, depth=depth, axis=1)            # (bz, depth)
        return self.cross_entropy_with_labelvec(prob, onehot_label, target=target, 
            weight=weight, sample_weight=sample_weight, neg_loss=neg_loss)

    def cross_entropy_with_labelvec(self, prob, label_vecs, target=1, weight=None, sample_weight=None, neg_loss=0, mask=None):
        """cross entropy with GT onehot vector"""
        assert target in [0,1]
        epsilon = 1e-8
        gamma = self.args.focal_loss
        alpha = neg_loss

        zeros = array_ops.zeros_like(label_vecs, dtype=prob.dtype) # (bz, depth)
        ones = array_ops.ones_like(label_vecs, dtype=prob.dtype) # (bz, depth)

        if target == 0:
            prob = 1-prob

        pos_position = label_vecs > array_ops.zeros_like(label_vecs, dtype=label_vecs.dtype)
        pos_prob = array_ops.where(pos_position, prob, ones)  # pos -> p, neg -> 1 (no loss)
        neg_prob = array_ops.where(pos_position, zeros, prob)  # pos -> 0(no loss), neg ->p


        pos_xent = - tf.log(tf.clip_by_value(pos_prob, epsilon, 1.0))
        if gamma is not None:
            pos_xent = ((1-pos_prob) ** gamma) * pos_xent
        
        if alpha is None or alpha == 0:
            neg_xent = 0
        else:
            neg_xent = - alpha * tf.log(tf.clip_by_value(1.0 - neg_prob, epsilon, 1.0))

            if gamma is not None:
                neg_xent = (neg_prob ** gamma) * neg_xent
        
        xent = pos_xent + neg_xent   # (bz, depth)

        if mask is not None:
            xent = xent*mask

        if sample_weight is not None:
            xent = xent*tf.expand_dims(sample_weight, axis=1)
        xent = tf.reduce_mean(xent, axis=0)  # (depth,)

        if weight is not None:
            xent = xent*weight

        return tf.reduce_sum(xent)
    


    def distance_metric(self, a, b):
        if (self.args.distance_metric == 'L2'):
            return tf.norm(a-b, axis=-1)
        elif (self.args.distance_metric == 'L1'):
            return tf.norm(a-b, axis=-1, ord=1)
        elif (self.args.distance_metric == 'cos'):
            return tf.reduce_sum(
                tf.multiply(
                    tf.nn.l2_normalize(a,axis=-1), 
                    tf.nn.l2_normalize(b,axis=-1)
                ), axis=-1)
        else:
            raise NotImplementedError("Unsupported distance metric: %s" + \
                self.args.distance_metric)
        
    def MSELoss(self, a, b):
        return tf.reduce_mean(self.distance_metric(a, b))

    def carlibration_look_up(self, prob, transform_attr_onehot, gt=1, margin = 0.05):
        if gt == 1:
            prob_label_vec = self.att_sim
        elif gt == 0:
            prob_label_vec = 1 - self.att_sim
        else:
            raise ValueError(gt)
        if np.any(prob_label_vec < 0):
            prob_label_vec = prob_label_vec * 0.5 + 0.5
        carlib_prob = tf.matmul(transform_attr_onehot, tf.cast(prob_label_vec, dtype=tf.float32))
        prob_diff = tf.abs(prob - carlib_prob)
        loss = tf.maximum(0., prob_diff - margin)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
        return loss

    def triplet_margin_loss(self, anchor, positive, negative, weight=None, margin=None):
        d1 = self.distance_metric(anchor,positive)
        d2 = self.distance_metric(anchor, negative)
        return self.triplet_margin_loss_with_distance(d1, d2, weight, margin)


    def triplet_margin_loss_with_distance(self, d1, d2, weight=None, margin=None):
        if margin is None:
            margin = self.args.triplet_margin
        if weight is None:
            dist = d1 - d2 + margin
        else:
            dist = weight * (d1 - d2) + margin
        return tf.maximum(dist, 0)
    
    def test_step(self, sess, blobs, score_op):
        dset = self.dataloader.dataset
        test_att = np.array([dset.attr2idx[attr] for attr, _ in dset.pairs])
        test_obj = np.array([dset.obj2idx[obj] for _, obj in dset.pairs])

        feed_dict = {
            self.pos_image_feat: blobs[4],
            self.test_attr_id: test_att,
            self.test_obj_id:  test_obj,
        }
        if self.args.obj_pred is not None:
            feed_dict[self.pos_obj_prediction] = blobs[-1]

        score = sess.run(score_op, feed_dict=feed_dict)

        for key in score_op.keys():
            score[key][0] = {
                (a,o): torch.from_numpy(score[key][0][:,i])
                for i,(a,o) in enumerate(zip(test_att, test_obj))
            }

        return score