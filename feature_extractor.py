import numpy as np
from numpy import linalg as LA
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
class ResNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model_resnet = ResNet50(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model_resnet.predict(np.zeros((1, 224, 224, 3)))
    '''
    Use Resnet model to extract features
    Output normalized feature vector
    '''
    #提取resnet50最后一层卷积特征
    def resnet_extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input_resnet(img)
        feat = self.model_resnet.predict(img)
        # print(feat.shape)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat
