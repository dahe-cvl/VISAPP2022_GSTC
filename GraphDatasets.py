import torch
from torch_geometric.data import Dataset, Data
import random
import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph


########################################
### CUSTOM DATASET
########################################
class GraphDataset(Dataset):
    def __init__(self, root, db_set=None, k_neighbors=500, threshold=0.5, use_distances=True, use_deprecated_version=False, split_factor=0.2, activate_masks=False, transform=None, pre_transform=None):
        self.db_set = db_set
        self.root = root
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.use_distances = use_distances
        self.split_factor = split_factor
        self.activate_masks = activate_masks
        self.use_deprecated_version = use_deprecated_version

        super(GraphDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        raw_path = os.path.join(self.root, "raw")
        filenames = os.listdir(raw_path)
        filenames.sort()
        return filenames

    @property
    def processed_file_names(self):
        return "not_implemented.pt"

    @property
    def num_classes(self):
        r"""The number of classes in the dataset."""
        y = self.data.y

        return len(torch.unique(y.squeeze()))

    def download(self):
        pass
        
    def process(self):
        created_files = os.listdir(self.processed_dir)
        if (self.use_deprecated_version == False):
            curr_file_name = 'data_k_{}.pt'.format(self.k_neighbors)
        else:
            curr_file_name = 'data_k_{}_deprecated.pt'.format(self.k_neighbors)

        #print(created_files)
        #print(curr_file_name)
        if(curr_file_name in created_files):
            print(f'Graph already created with k={self.k_neighbors} --> skip processing step')

            if (self.use_deprecated_version == False):
                self.data = torch.load(os.path.join(self.processed_dir, 'data_k_{}.pt'.format(self.k_neighbors)))
            else:
                self.data = torch.load(os.path.join(self.processed_dir, 'data_k_{}_deprecated.pt'.format(self.k_neighbors)))

            if(self.activate_masks == True):
                # make masks
                labels_np = self.data.y.detach().cpu().numpy()
                train_mask, test_mask = self._create_masks(labels_np)
                self.data.train_mask = train_mask
                self.data.test_mask = test_mask  

            if(self.transform != None):
                self.data = self.transform(self.data)
            #print(data) 
        else:
            print(f'The graph has to be created new with k={self.k_neighbors}')

            i = 0
            filename_l = []
            label_l = []
            features_l = []

            for sample_file in tqdm(self.raw_paths):
                db_info_np = np.load(sample_file)
                #print(db_info_np)
                #print(type(db_info_np[1]))
                #print(db_info_np[1].dtype)
                #exit()
                filename = db_info_np[0]
                label = db_info_np[1].astype('int')
                
                feature = db_info_np[2:].astype('float32')            
                filename_l.append(filename)
                label_l.append(label)
                features_l.append(feature)

            filenames_np = np.array(filename_l)
            labels_np = np.array(label_l)
            features_np = np.array(features_l)
            #print(filenames_np.shape)
            #print(labels_np.shape)
            #print(features_np.shape)

            features_tensor = torch.tensor(features_np, dtype=torch.float)
            labels_tensor = torch.tensor(labels_np, dtype=torch.long)
            edge_index, edge_attr = self._get_adjacency_info_NEW(features_np, vis_save_flag=True, threshold=self.threshold, return_distances=self.use_distances)

            if(self.activate_masks == True):
                # make masks
                train_mask, test_mask = self._create_masks(labels_np)
                self.data = Data(x=features_tensor,
                                y=labels_tensor,
                                edge_index=edge_index,
                                train_mask=train_mask, 
                                test_mask=test_mask,
                                edge_attr=edge_attr
                                )
            else:
                self.data = Data(x=features_tensor,
                                y=labels_tensor,
                                edge_index=edge_index,
                                edge_attr=edge_attr
                                )

            if(self.transform != None):
                self.data = self.transform(self.data)
            #print(data)
           
            if (self.use_deprecated_version == False):
                torch.save(self.data, os.path.join(self.processed_dir, 'data_k_{}.pt'.format(self.k_neighbors)))
            else:
                torch.save(self.data, os.path.join(self.processed_dir, 'data_k_{}_deprecated.pt'.format(self.k_neighbors)))
            

    def _create_masks(self, label_idx_np):

        label_names_np = np.unique(label_idx_np)

        train_mask_idx_all_l = []
        test_mask_idx_all_l = []

        for i in range(0, len(label_names_np)):
            label_idx = label_names_np[i]
            idx = np.where(label_idx == label_idx_np)[0]

            randomassort = list(idx)
            random.shuffle(randomassort)
            max_train = math.floor(len(randomassort) * self.split_factor)

            train_mask_idx = torch.tensor(randomassort[:max_train])
            test_mask_idx = torch.tensor(randomassort[max_train:])

            train_mask_idx_all_l.extend(train_mask_idx)
            test_mask_idx_all_l.extend(test_mask_idx)

        train_mask_idx_all_tensor = torch.stack(train_mask_idx_all_l)
        test_mask_idx_all_tensor = torch.stack(test_mask_idx_all_l)

        train_mask = torch.zeros(len(label_idx_np)) 
        test_mask = torch.zeros(len(label_idx_np))
        train_mask.scatter_(0, train_mask_idx_all_tensor, 1)
        test_mask.scatter_(0, test_mask_idx_all_tensor, 1)
        train_mask = train_mask.type(torch.bool)
        test_mask = test_mask.type(torch.bool)

        return train_mask, test_mask

    def _get_adjacency_info_NEW(self, features_np, vis_save_flag=False, return_distances=True, threshold=0.92):        
        adj_matrix = kneighbors_graph(features_np, n_neighbors=self.k_neighbors, mode='distance', include_self=True, n_jobs=4) #, metric="euclidean" 
        adj_matrix = adj_matrix.toarray()
        np.fill_diagonal(adj_matrix, 1)
        row, col = np.where(adj_matrix)
        np.fill_diagonal(adj_matrix, 0)
        coo = np.array(list(zip(row, col)))
        distance = adj_matrix[row, col]
        distance = np.expand_dims(distance, axis=1)
        
        if(vis_save_flag == True):
            #print(np.max(adj_matrix))
            #print(np.min(adj_matrix))
            #plt.figure()
            #plt.imshow(adj_matrix, cmap='gray')
            plt.figure()
            plt.imshow(adj_matrix[:300, :300], cmap='gray')
            #plt.show()
            plt.tight_layout()
            plt.savefig("./figure_" + str(features_np.shape[0]) + "_k_" + str(self.k_neighbors) + ".pdf")

        if (self.use_deprecated_version == False):
            coo = coo.T
        else:
            coo = np.reshape(coo, (2, -1))
        print(coo.shape)
        #coo = np.concatenate((coo, distance), axis=0)
        return torch.tensor(coo, dtype=torch.long), torch.tensor(distance, dtype=torch.float32)

    def _test_method(self):
        features_np = np.random.random((100, 2048))
        self._get_adjacency_matrix_cust(features_np, vis_save_flag=False, threshold=0.92)

    def len(self):
        return len(self.data)

    def get(self, idx):
        if (self.use_deprecated_version == False):
            data = torch.load(os.path.join(self.processed_dir, 'data_k_{}.pt'.format(self.k_neighbors)))
        else:
            data = torch.load(os.path.join(self.processed_dir, 'data_k_{}_deprecated.pt'.format(self.k_neighbors)))

        if(self.activate_masks == True):
            # make masks
            labels_np = data.y.detach().cpu().numpy()
            train_mask, test_mask = self._create_masks(labels_np)
            data.train_mask = train_mask
            data.test_mask = test_mask   
        return data
