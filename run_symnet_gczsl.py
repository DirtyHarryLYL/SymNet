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
from utils.evaluator import GCZSL_Evaluator

from run_symnet import make_parser as make_basic_parser
from run_symnet import SolverWrapper as CZSLSolverWrapper

def make_parser():
    parser = make_basic_parser()
    parser.add_argument("--test_set", type=str, default='val',
        choices=['test','val'])
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--bias", type=float, default=0)
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
    test_dataloader = dataset.get_dataloader(args.data, args.test_set, 
        batchsize=args.test_bz, obj_pred=args.obj_pred)
    


    logger.info("Loading network and solver")
    network = importlib.import_module('network.'+args.network)
    net = network.Network(train_dataloader, args)
    

    with utils.create_session() as sess:
        sw = SolverWrapper(net, train_dataloader, test_dataloader, args)
        sw.trainval_model(sess, args.epoch)




################################################################################



class SolverWrapper(CZSLSolverWrapper):
        

    def trainval_model(self, sess, max_epoch):
        logger = self.logger('train_model')
        logger.info('Begin training')

        lr, score_op, train_op, train_summary_op = self.construct_graph(sess)
        #for x in tf.global_variables():
        #    print(x.name)

        self.initialize(sess)
        sess.graph.finalize()


        writer = tf.summary.FileWriter(self.log_dir, sess.graph)
        evaluator = GCZSL_Evaluator(self.test_dataloader.dataset)

        best_report = defaultdict(dict)
        score_history = []

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
                    report = self.test(epoch, evaluator, value, all_attr_lab, all_obj_lab)

                    # save to tensorboard
                    summary = tf.Summary()
                    for k,v in report.items():
                        if k not in ['name', 'epoch']:
                            summary.value.add(tag="%s/%s"%(key,k), simple_value=v)
                    writer.add_summary(summary, epoch)



            if epoch % self.args.snapshot_freq == 0:
                self.snapshot(sess, epoch)


        writer.close()
        logger.info('Finished.')



    def test(self, epoch, evaluator, all_pred, all_attr_lab, all_obj_lab):
        args = self.args

        accuracies = []
        pairs = self.test_dataloader.dataset.pairs
        objs = self.test_dataloader.dataset.objs
        attrs = self.test_dataloader.dataset.attrs
        if args.test_set == 'test':
            val_pairs = self.test_dataloader.dataset.test_pairs
        else:
            val_pairs = self.test_dataloader.dataset.val_pairs
        train_pairs = self.test_dataloader.dataset.train_pairs
        

        all_attr_lab = torch.cat(all_attr_lab)
        all_obj_lab = torch.cat(all_obj_lab)
        all_pair_lab = torch.LongTensor([
            pairs.index((attrs[all_attr_lab[i]], objs[all_obj_lab[i]]))
            for i in range(len(all_attr_lab))
        ])
        all_pred_dict = {}
        for k in all_pred[0].keys():
            all_pred_dict[k] = torch.cat(
                [all_pred[i][k] for i in range(len(all_pred))])
        all_accuracies = []

        # Calculate best unseen acc
        # put everything on cpu
        attr_truth, obj_truth = all_attr_lab.cpu(), all_obj_lab.cpu()
        pairs = list(
            zip(list(attr_truth.cpu().numpy()), list(obj_truth.cpu().numpy())))
        seen_ind = torch.LongTensor([
            i for i in range(len(attr_truth))
            if pairs[i] in evaluator.train_pairs
        ])
        unseen_ind = torch.LongTensor([
            i for i in range(len(attr_truth))
            if pairs[i] not in evaluator.train_pairs
        ])




        # bias=0, for Causal-CZSl eval
        
        accuracies = []
        bias = 0
        args.bias = bias
        results = evaluator.score_model(
            all_pred_dict, all_obj_lab, bias=args.bias)
        match_stats = evaluator.evaluate_predictions(
            results, all_attr_lab, all_obj_lab, topk=args.topk)
        accuracies.append(match_stats)
        accuracies = zip(*accuracies)
        accuracies = list(map(torch.mean, map(torch.cat, accuracies)))
        attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
        print(
            '(val-causal) E:%d|A:%.3f|O:%.3f|Cl:%.3f|Op:%.4f|OpHM:%.4f|OpAvg:%.4f|OpSeen:%.4f|OpUnseen:%.4f|OrO:%.4f|maP:%.4f|bias:%.3f'
            % (
                epoch,
                attr_acc,
                obj_acc,
                closed_acc,
                open_acc,
                (open_seen_acc * open_unseen_acc)**0.5,
                0.5 * (open_seen_acc + open_unseen_acc),
                open_seen_acc,
                open_unseen_acc,
                objoracle_acc,
                0,
                bias,
            ))
        # end






        accuracies = []
        bias = 1e3
        args.bias = bias
        results = evaluator.score_model(
            all_pred_dict, all_obj_lab, bias=args.bias)
        match_stats = evaluator.evaluate_predictions(
            results, all_attr_lab, all_obj_lab, topk=args.topk)
        accuracies.append(match_stats)
        meanAP = 0
        _, _, _, _, _, _, open_unseen_match = match_stats
        accuracies = zip(*accuracies)
        open_unseen_match = open_unseen_match.byte()
        accuracies = list(map(torch.mean, map(torch.cat, accuracies)))
        attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
        scores = results['scores']
        correct_scores = scores[torch.arange(scores.shape[0]), all_pair_lab][
            unseen_ind]
        max_seen_scores = results['scores'][
            unseen_ind][:, evaluator.seen_mask].topk(
                args.topk, dim=1)[0][:, args.topk - 1]
        unseen_score_diff = max_seen_scores - correct_scores
        correct_unseen_score_diff = unseen_score_diff[open_unseen_match] - 1e-4
        full_unseen_acc = [(
            epoch,
            attr_acc,
            obj_acc,
            closed_acc,
            open_acc,
            (open_seen_acc * open_unseen_acc)**0.5,
            0.5 * (open_seen_acc + open_unseen_acc),
            open_seen_acc,
            open_unseen_acc,
            objoracle_acc,
            meanAP,
            bias,
        )]
        print(
            '(val) E:%d|A:%.3f|O:%.3f|Cl:%.3f|Op:%.4f|OpHM:%.4f|OpAvg:%.4f|OpSeen:%.4f|OpUnseen:%.4f|OrO:%.4f|maP:%.4f|bias:%.3f'
            % (
                epoch,
                attr_acc,
                obj_acc,
                closed_acc,
                open_acc,
                (open_seen_acc * open_unseen_acc)**0.5,
                0.5 * (open_seen_acc + open_unseen_acc),
                open_seen_acc,
                open_unseen_acc,
                objoracle_acc,
                meanAP,
                bias,
            ))

        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        biaslist = correct_unseen_score_diff[::bias_skip]

        for bias in biaslist:
            accuracies = []
            args.bias = bias
            results = evaluator.score_model(
                all_pred_dict, all_obj_lab, bias=args.bias)
            match_stats = evaluator.evaluate_predictions(
                results, all_attr_lab, all_obj_lab, topk=args.topk)
            accuracies.append(match_stats)
            meanAP = 0

            accuracies = zip(*accuracies)
            accuracies = map(torch.mean, map(torch.cat, accuracies))
            attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc, open_seen_acc, open_unseen_acc = accuracies
            all_accuracies.append((
                epoch,
                attr_acc,
                obj_acc,
                closed_acc,
                open_acc,
                (open_seen_acc * open_unseen_acc)**0.5,
                0.5 * (open_seen_acc + open_unseen_acc),
                open_seen_acc,
                open_unseen_acc,
                objoracle_acc,
                meanAP,
                bias,
            ))

            print(
                '(val) E:%d|A:%.3f|O:%.3f|Cl:%.3f|Op:%.4f|OpHM:%.4f|OpAvg:%.4f|OpSeen:%.4f|OpUnseen:%.4f|OrO:%.4f|maP:%.4f|bias:%.3f'
                % (
                    epoch,
                    attr_acc,
                    obj_acc,
                    closed_acc,
                    open_acc,
                    (open_seen_acc * open_unseen_acc)**0.5,
                    0.5 * (open_seen_acc + open_unseen_acc),
                    open_seen_acc,
                    open_unseen_acc,
                    objoracle_acc,
                    meanAP,
                    bias,
                ))
        all_accuracies.extend(full_unseen_acc)
        seen_accs = np.array([a[-5].item() for a in all_accuracies])
        unseen_accs = np.array([a[-4].item() for a in all_accuracies])
        area = np.trapz(seen_accs, unseen_accs)

        print(
            '(val) E:%d|A:%.3f|O:%.3f|Cl:%.3f|AUC:%.4f|Op:%.4f|OpHM:%.4f|OpAvg:%.4f|OpSeen:%.4f|OpUnseen:%.4f|OrO:%.4f|bias:%.3f'
            % (
                epoch,
                attr_acc,
                obj_acc,
                closed_acc,
                area,
                open_acc,
                (open_seen_acc * open_unseen_acc)**0.5,
                0.5 * (open_seen_acc + open_unseen_acc),
                open_seen_acc,
                open_unseen_acc,
                objoracle_acc,
                bias,
            ))

        #all_accuracies = [all_accuracies, area]
        #return all_accuracies

        return {
            "epoch":    epoch,
            "A":      attr_acc,
            "O":      obj_acc,
            "Cl":     closed_acc,
            "AUC":    area,
            "Op":     open_acc,
            "OpHM":   (open_seen_acc * open_unseen_acc)**0.5,
            "OpAvg":  0.5 * (open_seen_acc + open_unseen_acc),
            "OpSeen": open_seen_acc,
            "OpUnseen":open_unseen_acc,
            "OrO":    objoracle_acc,
            "bias":   bias,
        }





if __name__=="__main__":
    main()
