import numpy as np
import torch, torchvision
import os, logging, pickle, json
import tqdm
import os.path as osp

try:
    from . import data_utils
    from .. import config as cfg
except (ValueError, ImportError):
    import data_utils
    import config as cfg
from collections import defaultdict
from torch.utils.data import DataLoader

import argparse
import tqdm
import h5py


class MultiDatasetActivations(torch.utils.data.Dataset):
    def __init__(self, data, phase, args):
        super(MultiDatasetActivations, self).__init__()
        self.data = data
        self.phase = phase
        self.args = args

        
        pkl_path = './data/%s_%s.pkl' % (data, phase)
        logging.info("reading " + pkl_path)

        self.pkldata = pickle.load(open(pkl_path, "rb"))
        self.n_instance = len(self.pkldata)
        logging.info("# instances %d" % (self.n_instance))
        
        self.attrs = [a[0] for a in json.load(open('./utils/aux_data/%s_attrs.json' % data))]
        self.objs = [a[0] for a in json.load(open('./utils/aux_data/%s_objs.json' % data))]
        logging.info("# objs %d # attrs %d" % (len(self.objs), len(self.attrs)))

        self.gamma = json.load(open('./utils/aux_data/%s_gamma.json' % data))

        self.feat_dim = 2048

        self.att_sim_matrix = json.load(open(os.path.join(cfg.ROOT_DIR, "utils/aux_data/similarity_%s.json" % data)))
        self.att_sim_matrix = np.array(self.att_sim_matrix, dtype=np.float32)
        logging.info("generating neg/pos pool")

        

        # self.attr_pair[ i(i-1)/2 + j]=[i,j] i > j
        self.attr_pair = []
        for i in range(len(self.attrs)):
            for j in range(i):
                self.attr_pair.append([i, j])
        self.attr_pair = np.array(self.attr_pair, dtype=np.int32)

        self.neg_pool = []
        self.pos_pool = []
        self.img_att_sim = []
        self.neg_pair_filtered = []
        self.neg_pair_filtered_mask = []

        for i, info in enumerate(self.pkldata):
            self.neg_pool.append([])
            self.pos_pool.append([])

            for idx, label in enumerate(info['attr']):
                if label == 0:
                    self.neg_pool[i].append(idx)
                else:
                    self.pos_pool[i].append(idx)

            pos_sim = self.att_sim_matrix[np.array(info['attr'], dtype=np.bool)]
            if pos_sim.shape[0] == 0:
                self.img_att_sim.append(np.zeros(len(self.attrs)))
            else:
                self.img_att_sim.append(np.sum(pos_sim, axis=0))

            sort_idx = np.argsort(self.img_att_sim[i])[::-1]
            sort_idx = sort_idx[np.logical_not(np.array(info['attr'], dtype=np.bool))[sort_idx]]

            corre_idx = sort_idx[:int(len(sort_idx) * self.args.sample_bar)]
            neutral_idx = sort_idx[int(len(sort_idx) * (0.5 - 0.5 * self.args.sample_bar)): int(
                len(sort_idx) * (0.5 + 0.5 * self.args.sample_bar))]
            exclu_idx = sort_idx[int(len(sort_idx) * (1 - self.args.sample_bar)):]


            self.neg_pair_filtered.append([])
            self.neg_pair_filtered_mask.append(np.zeros(self.attr_pair.shape[0], dtype=np.bool))
            for a in neutral_idx:
                for b in corre_idx:
                    self.neg_pair_filtered[i].append([a, b])
                    mx = max(a, b)
                    mn = min(a, b)
                    self.neg_pair_filtered_mask[i][mx * (mx - 1) / 2 + mn] = True
                for b in exclu_idx:
                    self.neg_pair_filtered[i].append([a, b])
                    mx = max(a, b)
                    mn = min(a, b)
                    self.neg_pair_filtered_mask[i][mx * (mx - 1) / 2 + mn] = True



    def __getitem__(self, item):
        feature = self.pkldata[item]['feature']
        attr = self.pkldata[item]['attr']
        obj = self.pkldata[item]['category_id']
        img_a_sim = self.img_att_sim[item]
        random_attr_trip = np.random.permutation(len(self.attrs))
        pair_mask = self.neg_pair_filtered_mask[item]
        return feature, attr, obj, img_a_sim, random_attr_trip, pair_mask


    def __len__(self):
        return self.n_instance


