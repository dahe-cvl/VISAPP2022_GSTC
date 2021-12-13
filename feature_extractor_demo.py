from Dataloader import Dataset, ImageFolderWithPaths
from Models import FeatureExtractor

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn import metrics
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import torchvision
from matplotlib import pyplot as plt
import cv2

print("Welcome to the relation detection framework")

#db_path = "/data/share/datasets/vhh_rd_database_v1/"
#db_path = "/data/share/datasets/vhh_rd_database_v2/"
db_path = "/data/ext/VHH/datasets/public_datasets/caltech101/test/"
#db_path = "/data/ext/VHH/datasets/public_datasets/places365_standard/val/"
#db_path = "/data/ext/VHH/datasets/public_datasets/irsgs_scene_graph_similarity_db/"

SAVE_FEATURES_FLAG = True
batch_size = 64
num_workers = 2
backbone = "siamesenet_backbone_resnet152"
db_name = "caltech101"


class ToRGB(object):
    def __call__(self, img_np: np.ndarray):
        img_np = np.array(img_np)
        img_rgb_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        return img_rgb_np

    def __repr__(self):
        return self.__class__.__name__ + '_rgb_'

transform_test = transforms.Compose([
    #transforms.CenterCrop((250,250)),
    transforms.Resize((224, 224)),
    #ToRGB(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
])

'''
# load relation detection dataset
test_data = Dataset(path="/data/share/datasets/vhh_rd_database_v1/",
                    shuffle=False,
                    transform=transform_test)
testloader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )
'''

'''
test_data = torchvision.datasets.CIFAR10(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transform_test)
testloader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )
'''

'''
test_data = torchvision.datasets.FashionMNIST(root='./data/',
                                              train=True,
                                              download=True,
                                              transform=transform_test)
testloader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )
'''


# load relation detection dataset
test_data = ImageFolderWithPaths(root=db_path,
                                 transform=transform_test)
testloader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )

extractor = FeatureExtractor(db_name=db_name,
                             dataloader=testloader,
                             backbone=backbone,
                             save_features_as_numpy_flag=True)
#extractor.visualize_random_data_samples()
#exit()

extractor.get_all_features()
