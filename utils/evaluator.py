import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np
from . import utils
import itertools
import math
import collections
import logging
import sklearn.metrics as sklmetric


class CZSL_Evaluator:
    """modified from AttrOperator"""

    def __init__(self, dset, model):

        self.dset = dset 

        # convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) 
                 for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs)

        # mask over pairs that occur in closed world 
        test_pair_set = set(dset.test_pairs)
        mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
        self.closed_mask = torch.ByteTensor(mask)

        # object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj==obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.ByteTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)


    # generate masks for each setting, mask scores, and get prediction labels    
    def generate_predictions(self, scores, obj_truth): # (B, #pairs)

        def get_pred_from_scores(_scores):
            _, pair_pred = _scores.max(1)
            attr_pred, obj_pred = self.pairs[pair_pred][:,0], self.pairs[pair_pred][:,1]
            return (attr_pred, obj_pred)  # attr/obj word id (not name)

        def get_pred_from_scores_and_mask_best(_scores):
            _, pair_pred = _scores.max(1)
            attr_pred, obj_pred = self.pairs[pair_pred][:,0], self.pairs[pair_pred][:,1]
            _scores[range(pair_pred.shape[0]),pair_pred] = -1e10
            return _scores, (attr_pred, obj_pred)  # attr/obj word id (not name)

        results = {}

        # open world setting -- no mask
        results.update({'open': get_pred_from_scores(scores)})


        # closed world setting - set the score for all NON test-pairs to -1e10
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        if hasattr(mask, 'bool'):
            closed_scores[(1-mask).bool()] = -1e10
        else:
            closed_scores[(1-mask).byte()] = -1e10
        closed_scores, closed1 = get_pred_from_scores_and_mask_best(closed_scores)
        results.update({'closed1': closed1})
        closed_scores, closed2 = get_pred_from_scores_and_mask_best(closed_scores)
        results.update({'closed2': closed2})
        closed_scores, closed3 = get_pred_from_scores_and_mask_best(closed_scores)
        results.update({'closed3': closed3})


        # object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        if hasattr(mask, 'bool'):
            oracle_obj_scores[(1-mask).bool()] = -1e10
        else:
            oracle_obj_scores[(1-mask).byte()] = -1e10

        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores)})

        return results

    def score_model(self, scores, obj_truth):

        # put everything on CPU
        #scores = {k:v.cpu() for k,v in scores.items()}
        #obj_truth = obj_truth.cpu()

        # gather scores for all relevant (a,o) pairs
        scores = torch.stack([
            scores[(self.dset.attr2idx[attr], self.dset.obj2idx[obj])]
            for attr, obj in self.dset.pairs
        ], 1) # (B, #pairs)
        results = self.generate_predictions(scores, obj_truth)
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, histogram=False, synonym_mode=False):
        assert not histogram
  
        # put everything on cpu
        #attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()

        # top 1 pair accuracy
        # open world: attribute, object and pair
        attr_match = (attr_truth==predictions['open'][0]).float()
        obj_match = (obj_truth==predictions['open'][1]).float()
        open_match = attr_match*obj_match

        # closed world, obj_oracle: pair
        closed_1_match = (attr_truth==predictions['closed1'][0]).float() * (obj_truth==predictions['closed1'][1]).float()
        closed_2_match = (attr_truth==predictions['closed2'][0]).float() * (obj_truth==predictions['closed2'][1]).float() + closed_1_match
        closed_3_match = (attr_truth==predictions['closed3'][0]).float() * (obj_truth==predictions['closed3'][1]).float() + closed_2_match

        if synonym_mode:
            closed_2_match[closed_2_match>1] = 1
            closed_3_match[closed_3_match>1] = 1

        assert torch.max(closed_1_match).item()<=1, torch.max(closed_1_match).item()
        assert torch.max(closed_2_match).item()<=1, torch.max(closed_2_match).item()
        assert torch.max(closed_3_match).item()<=1, torch.max(closed_3_match).item()


        obj_oracle_match = (attr_truth==predictions['object_oracle'][0]).float() * (obj_truth==predictions['object_oracle'][1]).float()

        return attr_match, obj_match, closed_1_match, closed_2_match, closed_3_match, open_match, obj_oracle_match

    
    def evaluate_only_attr_obj(self, prob_a, gt_a, prob_o, gt_o):
        prob_a, prob_o = torch.from_numpy(prob_a), torch.from_numpy(prob_o)
        _, pred_a = prob_a.max(1)
        _, pred_o = prob_o.max(1)

        attr_match = (pred_a == gt_a).float()
        obj_match = (pred_o == gt_o).float()

        return attr_match, obj_match







class GCZSL_Evaluator:
    """modified from TMN"""

    def __init__(self, dset):

        self.dset = dset

        # convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # mask over pairs that occur in closed world
        if dset.phase == 'train':
            print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
        elif dset.phase == 'val':
            print('Evaluating with val pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
        else:
            print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
        self.test_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                           for attr, obj in list(test_pair_set)]
        mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
        self.closed_mask = torch.ByteTensor(mask)

        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.ByteTensor(mask)

        # object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.ByteTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)


    # generate masks for each setting, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth):  # (B, #pairs)
        def get_pred_from_scores(_scores):
            _, pair_pred = _scores.topk(10, dim=1)  #sort(1, descending=True)
            pair_pred = pair_pred[:, :10].contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, 10), self.pairs[pair_pred][:, 1].view(-1, 10)
            return (attr_pred, obj_pred)

        results = {}

        # open world setting -- no mask
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        mask = 1 - mask
        if hasattr(mask, "bool"):
            mask = mask.bool()
        closed_scores = scores.clone()
        closed_scores[mask] = -1e10
        results.update({'open': get_pred_from_scores(closed_scores)})

        # closed world setting - set the score for all NON test-pairs to -1e10
        #results.update({'closed': get_pred_from_scores(closed_scores)})
        results.update({'closed': results['open']})

        # object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        
        mask = 1 - mask
        if hasattr(mask, "bool"):
            mask = mask.bool()
        oracle_obj_scores[mask] = -1e10

        results.update({
            'object_oracle': get_pred_from_scores(oracle_obj_scores)
        })

        return results


    def score_model(self, scores, obj_truth, bias=0.0):
        # put everything on CPU
        scores = {k: v.cpu() for k, v in scores.items()}
        obj_truth = obj_truth.cpu()
        # gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(self.dset.attr2idx[attr], self.dset.obj2idx[obj])] for attr, obj in self.dset.pairs],
            1)  # (B, #pairs)
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        mask = 1 - mask
        if hasattr(mask, "bool"):
            mask = mask.bool()
        scores[mask] += bias
        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores
        results['scores'] = orig_scores
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth, topk=1):

        # put everything on cpu
        attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()
        pairs = list(
            zip(list(attr_truth.cpu().numpy()), list(obj_truth.cpu().numpy())))
        seen_ind = torch.LongTensor([
            i for i in range(len(attr_truth)) if pairs[i] in self.train_pairs
        ])
        unseen_ind = torch.LongTensor([
            i for i in range(len(attr_truth))
            if pairs[i] not in self.train_pairs
        ])

        # top 1 pair accuracy
        # open world: attribute, object and pair
        attr_match = (attr_truth.unsqueeze(1).repeat(
            1, topk) == predictions['open'][0][:, :topk])
        obj_match = (obj_truth.unsqueeze(1).repeat(
            1, topk) == predictions['open'][1][:, :topk])
        open_match = (attr_match * obj_match).any(1).float()
        attr_match = attr_match.any(1).float()
        obj_match = obj_match.any(1).float()
        open_seen_match = open_match[seen_ind]
        open_unseen_match = open_match[unseen_ind]

        # closed world, obj_oracle: pair
        closed_match = (attr_truth == predictions['closed'][0][:, 0]).float(
        ) * (obj_truth == predictions['closed'][1][:, 0]).float()

        obj_oracle_match = (
            attr_truth == predictions['object_oracle'][0][:, 0]).float() * (
                obj_truth == predictions['object_oracle'][1][:, 0]).float()

        return attr_match, obj_match, closed_match, open_match, obj_oracle_match, open_seen_match, open_unseen_match



class Multi_Evaluator:
    def __call__(self, prediction, gt_attr):
        assert prediction.shape == gt_attr.shape
        assert not np.any(np.isnan(prediction)), str(np.sum(np.isnan(prediction)))
        assert not np.any(np.isnan(gt_attr)), str(np.sum(np.isnan(gt_attr)))

        def calc_ap_auc(truth, scores):
            if np.sum(truth > 0) > 0:
                a = sklmetric.average_precision_score(truth, scores)
                b = sklmetric.roc_auc_score(truth, scores)
                assert not (np.isnan(a) or np.isnan(b))
                return a,b
            else:
                return np.nan,np.nan

        ap = np.zeros((gt_attr.shape[1],))
        auc = np.zeros((gt_attr.shape[1],))

        for dim in range(gt_attr.shape[1]):
            # rescale ground truth to [-1, 1]

            gt = gt_attr[:, dim]
            mask = (gt >= 0)

            gt = 2 * gt[mask] - 1  # = 0.5 threshold
            est = prediction[mask, dim]

            ap[dim],auc[dim] = calc_ap_auc(gt, est)

        mAP = np.nanmean(ap)
        mAUC = np.nanmean(auc)
        
        return mAP,mAUC
