import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from feature_extractor import ResNet
import sys
from h5py_generator import h5py_g

def predict(query,filepath,maxres):
    query = query
    index = 'resnet_featureCNN.h5'
    result = filepath
    maxres = maxres  # 检索出N张相似度最高的图片

    h5f = h5py.File(index, 'r')
    feats = h5f['dataset_1'][:]
    print(feats)
    imgNames = h5f['dataset_2'][:]
    print(imgNames)
    h5f.close()
    print("--------------------------------------------------")
    print("               searching starts")
    print("--------------------------------------------------")
    # read and show query image
    # queryDir = args["query"]
    queryImg = mpimg.imread(query)
    plt.title("Query Image")
    plt.imshow(queryImg)
    plt.show()

    model = ResNet()
    # extract query image's feature, compute simlarity score and sort
    queryVec = model.resnet_extract_feat(query)
    print(queryVec.shape)
    print(feats.shape)
    scores = np.dot(queryVec, feats.T)
    rank_ID = np.argsort(scores)[::-1]  # argsort函数返回的是数组值从小到大的索引值
    rank_score = scores[rank_ID]
    # print (rank_ID)
    print(rank_score)

    imlist = []
    for i, index in enumerate(rank_ID[0:maxres]):
        imlist.append(imgNames[index])
        # print(type(imgNames[index]))
        print("image names: " + str(imgNames[index]) + " scores: %f" % rank_score[i])
    print("top %d images in order are: " % maxres, imlist)
    # show top #maxres retrieved result one by one
    for i, im in enumerate(imlist):
        image = mpimg.imread(result + "/" + str(im, 'utf-8'))  # 以UTF-8的编码取得字节
        plt.title("Output %d" % (i + 1))
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    print("please enter absolute path of image to be searched for/请输入待查询图片的绝对路径")
    print("例如：/content/drive/My Drive/以图识图/家具")
    query = sys.stdin.readline().strip()

    print("please enter absolute path of database to be searched in/请输入图库的绝对路径")
    print("例如：/content/drive/My Drive/以图识图/2.jpg")
    filepath = sys.stdin.readline().strip()

    print("please enter the No. of most similar pics/请输入一个数，代表返回Top最相似的图片数量")
    maxres = int(sys.stdin.readline().strip())
    h5py_g(filepath)
    predict(query,filepath,maxres)