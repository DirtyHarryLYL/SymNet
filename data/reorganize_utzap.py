"""
Reorganize the UT-Zappos dataset to resemble the MIT-States dataset
root/attr_obj/img1.jpg
root/attr_obj/img2.jpg
root/attr_obj/img3.jpg
...
"""

import os
import torch
import glob
import shutil
import tqdm

root = 'ut-zap50k-original/'
os.makedirs(root+'/images')

data = torch.load(root+'/metadata.t7')
for instance in tqdm.tqdm(data):
	image, attr, obj = instance['_image'], instance['attr'], instance['obj']
	old_file = '%s/_images/%s'%(root, image)
	new_dir = '%s/images/%s_%s/'%(root, attr, obj)
	if not os.path.exists(new_dir):  # compatible to py2
		os.makedirs(new_dir)
	shutil.copy(old_file, new_dir)
