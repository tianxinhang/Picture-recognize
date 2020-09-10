import os
import h5py
import numpy as np
from feature_extractor import ResNet
import argparse
def h5py_g(filepath):
    # 搜索文件夹
    filepath=filepath  #'/content/drive/My Drive/以图识图/家具/'
    index = 'resnet_featureCNN.h5'
    feats = []
    names = []
    model = ResNet()
    for parent, dirnames, filenames in os.walk(filepath):
        for i,filename in enumerate(filenames):
            # print(filepath+filename)
            img_path=filepath+"/"+filename
            norm_feat = model.resnet_extract_feat(img_path)
            img_name = os.path.split(img_path)[1]
            print(img_name)
            feats.append(norm_feat)
            names.append(img_name)

    feats = np.array(feats)
    output = index
    h5f = h5py.File(output,'w')
    h5f.create_dataset('dataset_1', data = feats)
    h5f.create_dataset('dataset_2', data = np.string_(names))
    h5f.close()