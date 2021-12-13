from Dataloader import ImageFolderWithPaths
from Models import FeatureExtractor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import torch
from matplotlib import pyplot as plt
import cv2
import os
torch.manual_seed(17)

#db_path = "/caa/Projects02/vhh/private/database_nobackup/stc_vhh_mmsi_1_3_0/stc_vhh_mmsi_v1_3_0/"
db_path = "/caa/Projects02/vhh/private/database_nobackup/public_datasets/movienet/movienet_shottypes_split/"
#db_path = "/data/ext/VHH/datasets/public_datasets/cinescale/all_cinescale/"
#db_path = "/caa/Projects02/vhh/private/database_nobackup/all_cinescale/"

subset = "test"   # train val test all
SAVE_FEATURES_FLAG = True
SAVE_FEATURES_AS_SINGLE_SHOTS_FLAG = True
batch_size = 64
num_workers = 4
backbone = "movienetresnet50"  # resnet152 resnet18  vgg16  resnet152_gcn  stcresnet50  stc_vgg16 resnet50  movienetresnet50
db_name = "movienet_shottypes_split"  # all_cinescale  caltech101 places365 vhh_rd_database_v2 ucmerced  vhh_mmsi_v1_3_0_relation_db    stcv4   StcV4Graph_Resnet152  stc_vhh_mmsi_v1_3_0 movienet_shottypes_v2  all_movienet_shottypes_v2 movienet_shottypes_split histshotds
dst_path = "./extracted_features/"

db_path = os.path.join(db_path, subset)

class ToRGB(object):
    def __call__(self, img_np: np.ndarray):
        img_np = np.array(img_np)
        img_rgb_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        return img_rgb_np

    def __repr__(self):
        return self.__class__.__name__ + '_rgb_'

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([75.67079594559294 / 255.0,
                                68.76940725676867 / 255.0,
                                62.73719133427122 / 255.0],
                             [66.50369750799024 / 255.0,
                                64.23437522274287 / 255.0,
                                62.36074514298541 / 255.0
                            ])
])

test_data = ImageFolderWithPaths(root=db_path,
                                 transform=transform_test)

testloader = DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers
                        )

extractor = FeatureExtractor(dst_path=dst_path,
                             db_name=db_name,
                             dataloader=testloader,
                             backbone=backbone,
                             save_features_as_numpy_flag=SAVE_FEATURES_FLAG,
                             save_features_as_single_shot_flag=SAVE_FEATURES_AS_SINGLE_SHOTS_FLAG,
                             )

extractor.get_all_features()
