"""Modified from attributes-as-operators"""
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



class CompositionDatasetActivations(torch.utils.data.Dataset):

    def __init__(self, name, root, phase, feat_file, split='compositional-split', with_image=False, obj_pred=None, transform_type='normal'):
        self.root = root
        self.phase = phase
        self.split = split
        self.with_image = with_image

        self.feat_dim = None
        self.transform = data_utils.imagenet_transform(phase, transform_type)
        self.loader = data_utils.ImageLoader(self.root+'/images/')


        feat_file = os.path.join(root, feat_file)
        activation_data = torch.load(feat_file)

        self.activation_dict = dict(zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)
        print ('%d activations loaded'%(len(self.activation_dict)))

        # pair = (attr, obj)
        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}


        self.train_data, self.test_data = self.get_split_info()
        
        self.data = self.train_data if self.phase=='train' else self.test_data   # list of [img_name, attr, obj, attr_id, obj_id, feat]
        print ('#images = %d'%len(self.data))
        
        
        # return {object: all attrs that occur with obj}
        self.obj_affordance_mask = []
        for _obj in self.objs:
            candidates = [attr for (_,attr,obj,_,_,_) in self.train_data+self.test_data if obj==_obj]
            affordance = set(candidates)
            mask = [1 if x in affordance else 0   for x in self.attrs]
            self.obj_affordance_mask.append(mask)
        

        # negative image pool
        samples_grouped_by_obj = [[] for _ in range(len(self.objs))]
        for i,x in enumerate(self.train_data):
            samples_grouped_by_obj[x[4]].append(i)

        self.neg_pool = []  # [obj_id][attr_id] => list of sample id
        for obj_id in range(len(self.objs)):
            self.neg_pool.append([])
            for attr_id in range(len(self.attrs)):
                self.neg_pool[obj_id].append(
                    [i for i in samples_grouped_by_obj[obj_id] if 
                        self.train_data[i][3] != attr_id ]
                )

        aux_data_root = osp.join(cfg.ROOT_DIR, 'utils/aux_data')
        gamma = json.load(open(aux_data_root + "/%s_gamma.json"%name))
        gamma = {k:np.array(v, dtype=np.float32) for k,v in gamma.items()}
        self.comp_gamma = {'a':gamma['comp_a'], 'b':gamma['comp_b']}
        self.attr_gamma = {'a':gamma['attr_a'], 'b':gamma['attr_b']}
        

        if obj_pred is None:
            self.obj_pred = None
        else:
            obj_pred_path = osp.join(cfg.ROOT_DIR, 'data/obj_scores', obj_pred)
            print("Loading object prediction from %s"%obj_pred_path.split('/')[-1])
            with open(obj_pred_path, 'rb') as fp:
                self.obj_pred = np.array(pickle.load(fp), dtype=np.float32)

        
    def get_split_info(self):
        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        test_pair_set = set(self.test_pairs)
        train_data, test_data = [], []

        for instance in data:

            image, attr, obj = instance['image'], instance['attr'], instance['obj']

            if attr=='NA' or (attr, obj) not in self.pairs:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj, self.attr2idx[attr], self.obj2idx[obj], self.activation_dict[image]]
            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, test_data

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))

        all_attrs, all_objs =  sorted(list(set(tr_attrs+ts_attrs))), sorted(list(set(tr_objs+ts_objs)))    
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs


    def sample_negative(self, attr_id, obj_id):
        return np.random.choice(self.neg_pool[obj_id][attr_id])


    def __getitem__(self, index):
        def get_sample(i):
            image, attr, obj, attr_id, obj_id, feat = self.data[i]
            if self.with_image:
                img = self.loader(image)
                img = self.transform(img)
            else:
                img = None

            return [img, attr_id, obj_id, self.pair2idx[(attr, obj)], feat]

        pos = get_sample(index)

        mask = np.array(self.obj_affordance_mask[pos[2]], dtype=np.float32)


        if self.phase=='train':
            negid = self.sample_negative(pos[1], pos[2]) # negative example
            neg = get_sample(negid)

            data = pos + neg + [mask]
        else:
            data = pos + [mask]

        # train [img, attr_id, obj_id, pair_id, img_feature, img, attr_id, obj_id, pair_id, img_feature, aff_mask]
        # test [img, attr_id, obj_id, pair_id, img_feature, aff_mask]

        if self.obj_pred is not None:
            data.append(self.obj_pred[index,:])
        
        return data

    def __len__(self):
        return len(self.data)
    






class CompositionDatasetActivationsGenerator(CompositionDatasetActivations):

    def __init__(self, root, feat_file, split='compositional-split', feat_extractor=None, transform_type='normal'):
        super(CompositionDatasetActivationsGenerator, self).__init__(root, 'train', feat_file, split, transform_type=transform_type)

        assert os.path.exists(root)
        with torch.no_grad():
            self.generate_features(feat_file, feat_extractor, transform_type)
        print('Features generated.')
        

    def generate_features(self, out_file, feat_extractor, transform_type):

        data = self.train_data+self.test_data
        transform = data_utils.imagenet_transform('test', transform_type)

        if feat_extractor is None:
            feat_extractor = torchvision.models.resnet18(pretrained=True)
            feat_extractor.fc = torch.nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(data_utils.chunks(data, 512), total=len(data)//512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs))
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print ('features for %d images generated'%(len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)
    




if __name__=='__main__':
    """example code for generating new features for MIT states and UT Zappos
    CompositionDatasetActivationsGenerator(
        root = 'data-dir', 
        feat_file = 'filename-to-save', 
        feat_extractor = torchvision.models.resnet18(pretrained=True),
    )
    """
    CompositionDatasetActivationsGenerator(
        root = 'data/attributes-as-operators/data/mit-states',
        feat_file = 'data/attributes-as-operators/data/mit-states/features.t7',
    )