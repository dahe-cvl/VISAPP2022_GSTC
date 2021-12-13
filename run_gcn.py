import torch
import torch_geometric.transforms as T
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from torch_geometric.profile import get_cpu_memory_from_gc
from torch_geometric.data import Data
from torchvision import transforms
import numpy as np
import os
import csv 
import sys
import json
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from sklearn.metrics import classification_report, confusion_matrix

from GraphModels import GCN, GraphSage, GAT, GraphGIN, TransformerGAT
from GraphDatasets import GraphDataset
from Utils import calculate_tsne, calculate_umap, plot_confusion_matrix
from Logger import Logger



########################################
### TRAINING METHODS
########################################

def train(config_params_dict=None):
    print("run training")

    root_path = config_params_dict['root_path']  #
    subfolder = config_params_dict['subfolder']  #
    project_name = config_params_dict['project_name']  # "StcVhhMmsiGraph"  # StcV4Graph  Cifar10Graph Caltech101Graph  StcVhhMmsiGraph
    orig_db_name = config_params_dict['orig_db_name']  # "stc_vhh_mmsi_v1_3_0"       # cifar10  stcv4  caltech101 stc_vhh_mmsi_v1_3_0
    model_architecture = config_params_dict['model_architecture']  # "sage"
    backbone = config_params_dict['backbone']  # "Resnet152"         #  Resnet152_all Resnet152_train Resnet152_val Vgg16
    subset = config_params_dict['subset']  # "train"               # train test val
    n_epochs = config_params_dict['n_epochs']  # 1000
    batch_size = config_params_dict['batch_size']  # 1024
    num_workers = config_params_dict['num_workers']  # 4
    lr = config_params_dict['lr']  # 0.001
    weight_decay = config_params_dict['weight_decay']  # 0.08
    k_neighbors = config_params_dict['k_neighbors']  # 1000
    sampler_sizes_l = config_params_dict['sampler_sizes_l']  # [25, 15, 5]
    hidden_channels = config_params_dict['hidden_channels']  # 256
    early_stopping_threshold = config_params_dict['early_stopping_threshold'] #60
    lr_scheduler_patience = config_params_dict['lr_scheduler_patience'] #30
    threshold = config_params_dict['threshold'] #30
    use_distances = config_params_dict['use_distances'] #30
    split_factor = config_params_dict['split_factor'] #30
    use_deprecated_version = config_params_dict['use_deprecated_version']
    num_heads = config_params_dict['heads']
    dropout = config_params_dict['dropout']
    aggr_mode = config_params_dict['aggr_mode']

    model_name = project_name + "_" + backbone + "_" + model_architecture
    db_name = project_name + "_" + backbone
    experiment_name = backbone + "_" + model_architecture
    #db_path = "./data/" + db_name
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/resnet152/stcv4/" + db_name 
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/vgg16/stcv4/" + db_name     
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/" + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + db_name 
    db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + subset
    val_db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + "val"

    wb_logger = Logger()
    
    wb_logger.set_general_experiment_params(
        exp_config=config_params_dict,
        experiment_dir="/caa/Projects02/vhh/private/experiments_nobackup/graph_based_relations/",
        project_name=project_name,
        experiment_name=experiment_name
    )

    # create experiment folder
    print("create experiment folder...")
    time_stamp = datetime.now().strftime("%Y%m%d")
    experiment_name = str(time_stamp) + "_" + str(project_name) + "_" + str(orig_db_name) + "_" + str(model_architecture) + "_" + str(backbone)
    experiment_dir = "./results/" + experiment_name
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
 
    experiments_idx = len(os.listdir(experiment_dir)) + 1
    experiment_dir = experiment_dir + "/exp_" + str(experiments_idx)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    with open(experiment_dir + "/" + model_name + ".json", 'w') as fp:
        json.dump(config_params_dict, fp)

    wb_logger.initialize_logger()

    #dataset = Planetoid("./data/Cora", 'Cora', transform=T.NormalizeFeatures()) #

    transform = transforms.Compose([
                                    T.ToUndirected(),
                                   ])

    dataset = GraphDataset(root=db_path, 
                           db_set="all", 
                           activate_masks=True, 
                           transform=None, 
                           pre_transform=None, 
                           k_neighbors=k_neighbors, 
                           threshold=threshold,
                           use_distances=use_distances,
                           split_factor=split_factor,
                           use_deprecated_version=use_deprecated_version
                           )   #NormalizeFeatures()
    '''
    val_dataset = GraphDataset(root=val_db_path, 
                                db_set="all", 
                                activate_masks=True, 
                                transform=None, 
                                pre_transform=None, 
                                k_neighbors=k_neighbors, 
                                threshold=threshold,
                                use_distances=use_distances,
                                split_factor=split_factor,
                                use_deprecated_version=use_deprecated_version
                                )   #NormalizeFeatures()
    '''

    ''''''
    print(dataset.num_node_features)
    print(dataset.num_features)
    print(dataset.num_edge_features)
    data = dataset[0]

    train_idx = torch.where(data.train_mask == 1)[0]
    val_idx = torch.where(data.test_mask == 1)[0]
    #test_idx = torch.where(data.test_mask == 1)[0]
    #train_idx = torch.arange(data.x.size(0))
    print(len(train_idx))
    print(len(val_idx))

    '''
    print(val_dataset.num_node_features)
    print(val_dataset.num_features)
    print(val_dataset.num_edge_features)
    val_data = val_dataset[0]
    val_idx = torch.where(val_data.train_mask == 1)[0]
    #val_test_idx = torch.where(val_data.test_mask == 1)[0]
    #val_idx = torch.arange(val_data.x.size(0))
    print(len(val_idx))
    #print(len(val_test_idx))
    '''
    
    '''
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx,
                                sizes=sampler_sizes_l, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)


    val_loader = NeighborSampler(data.edge_index, node_idx=test_idx,
                                sizes=sampler_sizes_l, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)            
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = NeighborLoader(data,
                                  num_neighbors=sampler_sizes_l,
                                  batch_size=batch_size,
                                  input_nodes=train_idx,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  replace=True,
                                  directed=True,
                                  persistent_workers=False
                                 )
    
    val_loader = NeighborLoader(data,
                                num_neighbors=sampler_sizes_l,
                                batch_size=batch_size,
                                input_nodes=val_idx,
                                shuffle=False,
                                num_workers=num_workers,
                                replace=True,
                                directed=True,
                                persistent_workers=False
                                )


    
    print(data.num_nodes)
    print(data.num_edges)
    print(data.train_mask.sum())
    print(data.test_mask.sum())
    print(dataset.num_classes)
    '''
    print(val_data.num_nodes)
    print(val_data.num_edges)
    print(val_data.train_mask.sum())
    print(val_data.test_mask.sum())
    print(val_dataset.num_classes)
    '''

    if (model_architecture == "transformer"):
    
        model = TransformerGAT(in_channels=dataset.num_features, 
                                hidden_channels=hidden_channels, 
                                out_channels=dataset.num_classes, 
                                aggr_mode=aggr_mode,
                                dropout=dropout,
                                num_heads=num_heads,
                                use_edge_attr=use_distances,
                                return_embeddings_flag=True
                            ).to(device)

    elif (model_architecture == "gin"):
    
        model = GraphGIN(in_channels=dataset.num_features, 
                            hidden_channels=hidden_channels, 
                            out_channels=dataset.num_classes, 
                            aggr_mode=aggr_mode,
                            dropout=dropout,
                            return_embeddings_flag=True
                            ).to(device)

    elif(model_architecture == "sage"):
    
        model = GraphSage(in_channels=dataset.num_features, 
                          hidden_channels=hidden_channels, 
                          out_channels=dataset.num_classes, 
                          aggr_mode=aggr_mode,
                          dropout=dropout,
                          return_embeddings_flag=True
                        ).to(device)
    
    elif(model_architecture == "gcn"):

        model = GCN(in_channels=dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    return_embeddings_flag=True,
                    use_edge_attr=use_distances
                ).to(device)
    
    elif(model_architecture == "gat"):
    
        model = GAT(in_channels=dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    num_heads=num_heads,
                    return_embeddings_flag=True,
                    use_edge_attr=use_distances
                ).to(device)
    
    else:
        print("ERROR: select valid model architecture!")
        exit()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    print("[Creating Learning rate scheduler...]")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=0.000001, patience=lr_scheduler_patience, verbose=True)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, verbose=True)

    best_acc = 0
    best_loss = 1000
    early_stopping_cnt = 0

    wb_logger.log_model(model)
    for epoch in range(0, n_epochs):
        #data = data.to(device)
        #x = data.x.to(device)
        #y = data.y.squeeze().to(device)

        total_loss = 0 
        total_correct = 0
        model.train()
        for i, data_batch in enumerate(train_loader):
            data_batch = transform(data_batch)
            data_batch = data_batch.to(device)
            #print(data_batch.size())
            #print(data_batch)
            
            #if(i % 10 == 0):
            #    plot_dataset(data_batch, "graph_batch", experiment_dir=experiment_dir, idx=i)
            #    data_batch = data_batch.to(device)
            #exit()
            #continue
            
            optimizer.zero_grad()
            #out, embeddings = model.batch_forward(data_batch.x, data_batch.edge_index)
            out, embeddings = model.batch_forward(data_batch)

            loss = criterion(out[:data_batch.batch_size], data_batch.y[:data_batch.batch_size])
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out[:data_batch.batch_size].argmax(dim=-1).eq(data_batch.y[:data_batch.batch_size]).sum())
           
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / train_idx.size(0)
        del loss

        with torch.no_grad():
            total_loss = 0 
            total_correct = 0
            model.eval()

            for data_batch in val_loader:
                data_batch = transform(data_batch)
                data_batch = data_batch.to(device)
                out, embeddings = model.batch_forward(data_batch)
                #out, embeddings = model.batch_forward(data_batch.x, data_batch.edge_index)
                loss = criterion(out[:data_batch.batch_size], data_batch.y[:data_batch.batch_size])
                
                total_loss += float(loss)
                total_correct += int(out[:data_batch.batch_size].argmax(dim=-1).eq(data_batch.y[:data_batch.batch_size]).sum())

            val_loss = total_loss / len(val_loader)
            val_acc = total_correct / val_idx.size(0)

            del loss
        ###############################
        # Logging.
        ###############################
        print(f'Epoch: {epoch:03d}/{n_epochs}, Train Loss: {train_loss:.7f}, Train Acc: {train_acc:.7f}, Validation Loss: {val_loss:.7f}, Validation Acc: {val_acc:.7f}')
        metrics_dict = {
            "train_loss": train_loss, #.item(),
            "train_acc": train_acc,
            "val_loss": val_loss, #.item(),
            "val_acc": val_acc
        }
        wb_logger.log_metrics(metrics_dict)
        
        ###############################
        # Save checkpoint.
        ###############################
        acc_curr = val_acc
        vloss_curr = val_loss
        #if acc_curr > best_acc and vloss_curr < best_loss:
        #if vloss_curr < best_loss:
        if acc_curr > best_acc:
            print('Saving...')
            state = {
                'net': model.state_dict(),
                'acc': acc_curr,
                'loss': vloss_curr,
                'epoch': epoch,
            }
            # if not os.path.isdir('checkpoint'):
            #    os.mkdir('checkpoint')
            #torch.save(state, expFolder + "/" "best_model" + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
            torch.save(state, experiment_dir + "/" + model_name + ".pth")

            best_acc = acc_curr
            best_loss = vloss_curr
            early_stopping_cnt = 0
        
        scheduler.step(val_loss)

        ###############################
        # early stopping.
        ###############################
        #if (acc_curr <= best_acc):
        if (vloss_curr >= best_loss):
            early_stopping_cnt = early_stopping_cnt + 1
        if (early_stopping_cnt >= early_stopping_threshold):
            print('Early stopping active --> stop training ...')
            break

    # final evaluation on validation set 
    print("load pre-trained model ...")
    model_states = torch.load(experiment_dir + "/" + model_name + ".pth")
    model.load_state_dict(model_states['net'])
    print(model_states['loss'])
    print(model_states['acc'])
    print(model_states['epoch'])
   
    with torch.no_grad():
        total_loss = 0 
        total_correct = 0
        total_correct = 0
        all_preds = []
        all_labels = []
        all_embeddings = []
        model.eval()

        for data_batch in val_loader:
            data_batch = transform(data_batch)
            data_batch = data_batch.to(device)
            #out, embeddings = model.batch_forward(data_batch.x, data_batch.edge_index)
            out, embeddings = model.batch_forward(data_batch)

            if embeddings != None: 
                all_embeddings.extend(embeddings[:data_batch.batch_size])
            
            pred = out.argmax(dim=-1)
            total_correct += int(out[:data_batch.batch_size].argmax(dim=-1).eq(data_batch.y[:data_batch.batch_size]).sum())
            all_preds.extend(pred[:data_batch.batch_size])
            all_labels.extend(data_batch.y[:data_batch.batch_size])

        test_acc = total_correct / val_idx.size(0)
        print(f'validation accuracy: {test_acc:.4f}')   

        if embeddings != None: 
            all_embeddings_tensor = torch.stack(all_embeddings)
        all_preds_tensor = torch.stack(all_preds)
        all_labels_tensor = torch.stack(all_labels)
        if embeddings != None: 
            print(all_embeddings_tensor.size())
        print(all_preds_tensor.size())
        print(all_labels_tensor.size())

        # save classification report
        print("calculate classification report and confusion matrices ... ")
        classes = ["CU", "ELS", "I", "LS", "MS", "NA"]
        #classes = ["CS", "ECS", "FS", "LS", "MS"]
        x = all_preds_tensor.detach().cpu().numpy()
        y = all_labels_tensor.detach().cpu().numpy()

        report_dict = classification_report(y, x, target_names=classes, output_dict=True)
        matrix = confusion_matrix(y, x)

        
        with open(experiment_dir + "/report_val_" + model_name + ".json", 'w') as fp:
            json.dump(report_dict, fp)

        # save confusion matrix 
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=True,
                              path=experiment_dir + "/val_confusion_matrix_normalize.pdf")
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=False,
                              path=experiment_dir + "/val_confusion_matrix.pdf")
    
    if embeddings != None: 
        print("calculate tsne and umap plots ... ")
        embeddings_all_np = all_embeddings_tensor.detach().cpu().numpy()
        all_lables_np = y

        print(embeddings_all_np.shape)
        print(all_lables_np.shape)

        #print("save embeddings ... ")
        #np.save("./embeddings_all_np.npy", embeddings_all_np)
        #np.save("./all_lables_np.npy", all_lables_np)

        name = model_name + "_val"
        calculate_tsne(name, embeddings_all_np, all_lables_np, experiment_dir, target_names=classes)
        calculate_umap(name, embeddings_all_np, all_lables_np, experiment_dir, target_names=classes)
   
    wb_logger.finish_run()


########################################
### TEST METHODS
########################################

def test(config_params_dict=None, experiment_dir="./"):
    print("run test")

    root_path = config_params_dict['root_path']  #
    subfolder = config_params_dict['subfolder']  #
    project_name = config_params_dict['project_name']  # "StcVhhMmsiGraph"  # StcV4Graph  Cifar10Graph Caltech101Graph  StcVhhMmsiGraph
    orig_db_name = config_params_dict['orig_db_name']  # "stc_vhh_mmsi_v1_3_0"       # cifar10  stcv4  caltech101 stc_vhh_mmsi_v1_3_0
    model_architecture = config_params_dict['model_architecture']  # "sage"
    backbone = config_params_dict['backbone']  # "Resnet152"         #  Resnet152_all Resnet152_train Resnet152_val Vgg16
    subset = config_params_dict['subset']  # "train"               # train test val
    batch_size = config_params_dict['batch_size']  # 1024
    num_workers = config_params_dict['num_workers']  # 4
    k_neighbors = config_params_dict['k_neighbors']  # 1000
    sampler_sizes_l = config_params_dict['sampler_sizes_l']  # [25, 15, 5]
    hidden_channels = config_params_dict['hidden_channels']  # 256
    threshold = config_params_dict['threshold'] #30
    use_distances = config_params_dict['use_distances'] #30
    split_factor = config_params_dict['split_factor'] #30
    use_deprecated_version = config_params_dict['use_deprecated_version']
    num_heads = config_params_dict['heads']
    dropout = config_params_dict['dropout']
    aggr_mode = config_params_dict['aggr_mode']

    model_name = project_name + "_" + backbone + "_" + model_architecture
    db_name = project_name + "_" + backbone
    experiment_name = backbone + "_" + model_architecture
    #db_path = "./data/" + db_name
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/resnet152/stcv4/" + db_name 
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/vgg16/stcv4/" + db_name     
    #db_path = "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/" + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + db_name 

    db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + str(subset)
    dataset = GraphDataset(root=db_path, 
                           db_set="all", 
                           activate_masks=True, 
                           transform=None, 
                           pre_transform=None, 
                           k_neighbors=k_neighbors, 
                           threshold=threshold,
                           use_distances=use_distances,
                           split_factor=split_factor,
                           use_deprecated_version=use_deprecated_version
                           )   #NormalizeFeatures()

    print(dataset.num_node_features)
    print(dataset.num_features)
    print(dataset.num_edge_features)
    data = dataset[0]
    node_idx = torch.arange(data.num_nodes)
    print(len(node_idx))
    print(data.num_nodes)
    print(data.num_edges)
    print(data.test_mask.sum())
    print(dataset.num_classes)

    test_db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + "test"
    #dataset = Planetoid("./data/Cora", 'Cora', transform=T.NormalizeFeatures()) #
    test_dataset = GraphDataset(root=test_db_path, 
                           db_set="all", 
                           activate_masks=True, 
                           transform=None, 
                           pre_transform=None, 
                           k_neighbors=k_neighbors, 
                           threshold=threshold,
                           use_distances=use_distances,
                           split_factor=split_factor,
                           use_deprecated_version=use_deprecated_version
                           )   #NormalizeFeatures()
    ''''''
    print(test_dataset.num_node_features)
    print(test_dataset.num_features)
    print(test_dataset.num_edge_features)
    test_data = test_dataset[0]
    test_idx = torch.arange(test_data.num_nodes)
    print(len(test_idx))
    print(test_data.num_nodes)
    print(test_data.num_edges)
    print(test_data.test_mask.sum())
    print(test_dataset.num_classes)


    # combine test dataset with root database
    orig_features = data.x
    test_features = test_data.x

    orig_labels = data.y
    test_labels = test_data.y

    combined_features = torch.cat((orig_features, test_features), dim=0)
    print(combined_features.size())

    if(use_distances == False):
        combined_edges, combined_edge_attr = dataset._get_adjacency_info_NEW(combined_features.detach().cpu().numpy(), return_distances=use_distances)
        print(combined_edges.size())
        #combined_edge_attr = None
    else:
        combined_edges, combined_edge_attr = dataset._get_adjacency_info_NEW(combined_features.detach().cpu().numpy(), return_distances=use_distances)
        print(combined_edges.size())
        print(combined_edge_attr.size())

    combined_labels = torch.cat((orig_labels, test_labels), dim=0)
    print(combined_labels.size())

    all_data = Data(x=combined_features,
                    y=combined_labels,
                    edge_index=combined_edges,
                    train_mask=torch.ones((combined_features.size(0))),
                    test_mask=torch.ones((combined_features.size(0))),
                    edge_attr=combined_edge_attr
                   )
    print(all_data)

    all_idx = torch.arange(all_data.num_nodes)
    all_idx = all_idx[data.x.size(0):]   # data.x.size(0)
    print(len(all_idx))
   
    test_loader = NeighborLoader(all_data,
                                 num_neighbors=sampler_sizes_l,
                                 batch_size=batch_size,
                                 input_nodes=all_idx,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 replace=True,
                                 directed=True,
                                 persistent_workers=False
                                )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if (model_architecture == "transformer"):
    
        model = TransformerGAT(in_channels=test_dataset.num_features, 
                                hidden_channels=hidden_channels, 
                                out_channels=test_dataset.num_classes, 
                                aggr_mode=aggr_mode,
                                dropout=dropout,
                                num_heads=num_heads,
                                return_embeddings_flag=True,
                                use_edge_attr=use_distances
                            ).to(device)

    elif (model_architecture == "gin"):
    
        model = GraphGIN(in_channels=test_dataset.num_features, 
                            hidden_channels=hidden_channels, 
                            out_channels=test_dataset.num_classes, 
                            aggr_mode=aggr_mode,
                            dropout=dropout
                            ).to(device)

    elif(model_architecture == "sage"):
    
        model = GraphSage(in_channels=test_dataset.num_features, 
                          hidden_channels=hidden_channels, 
                          out_channels=test_dataset.num_classes, 
                          aggr_mode=aggr_mode,
                          dropout=dropout,
                          return_embeddings_flag=True
                        ).to(device)
    
    elif(model_architecture == "gcn"):

        model = GCN(in_channels=test_dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=test_dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    return_embeddings_flag=True,
                    use_edge_attr=use_distances
                ).to(device)
    
    elif(model_architecture == "gat"):
    
        model = GAT(in_channels=test_dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=test_dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    num_heads=num_heads,
                    return_embeddings_flag=True,
                ).to(device)
    
    else:
        print("ERROR: select valid model architecture!")
        exit()


    #wb_logger.log_model(model)

    model_states = torch.load(experiment_dir + "/" + model_name + ".pth")
    model.load_state_dict(model_states['net'])

    transform = transforms.Compose([
                                    T.ToUndirected(),
                                   ])

   
    with torch.no_grad():
        total_loss = 0 
        total_correct = 0
        total_correct = 0
        all_preds = []
        all_labels = []
        all_embeddings = []
        model.eval()

        for data_batch in test_loader:
            data_batch = transform(data_batch)
            data_batch = data_batch.to(device)
            #out, embeddings = model.batch_forward(data_batch.x, data_batch.edge_index)
            out, embeddings = model.batch_forward(data_batch)

            if embeddings != None: 
                all_embeddings.extend(embeddings[:data_batch.batch_size])
            
            pred = out.argmax(dim=-1)
            total_correct += int(out[:data_batch.batch_size].argmax(dim=-1).eq(data_batch.y[:data_batch.batch_size]).sum())
            all_preds.extend(pred[:data_batch.batch_size])
            all_labels.extend(data_batch.y[:data_batch.batch_size])

        test_acc = total_correct / all_idx.size(0)
        print(f'test accuracy: {test_acc:.4f}')   

        if embeddings != None: 
            all_embeddings_tensor = torch.stack(all_embeddings)
        all_preds_tensor = torch.stack(all_preds)
        all_labels_tensor = torch.stack(all_labels)
        if embeddings != None: 
            print(all_embeddings_tensor.size())
        print(all_preds_tensor.size())
        print(all_labels_tensor.size())

        # save classification report
        print("calculate classification report and confusion matrices ... ")
        classes = ["CU", "ELS", "I", "LS", "MS", "NA"]
        #classes = ["CS", "ECS", "FS", "LS", "MS"]
        x = all_preds_tensor.detach().cpu().numpy()
        y = all_labels_tensor.detach().cpu().numpy()

        report_dict = classification_report(y, x, target_names=classes, output_dict=True)
        matrix = confusion_matrix(y, x)

        with open(experiment_dir + "/report_test_" + model_name + ".json", 'w') as fp:
            json.dump(report_dict, fp)

        # save confusion matrix 
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=True,
                              path=experiment_dir + "/test_confusion_matrix_normalize.pdf")
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=False,
                              path=experiment_dir + "/test_confusion_matrix.pdf")
    
    if embeddings != None: 
        print("calculate tsne and umap plots ... ")
        embeddings_all_np = all_embeddings_tensor.detach().cpu().numpy()
        all_lables_np = y

        print(embeddings_all_np.shape)
        print(all_lables_np.shape)

        #print("save embeddings ... ")
        #np.save("./embeddings_all_np.npy", embeddings_all_np)
        #np.save("./all_lables_np.npy", all_lables_np)

        name = model_name + "_test"
        calculate_tsne(name, embeddings_all_np, all_lables_np, experiment_dir, target_names=classes)
        calculate_umap(name, embeddings_all_np, all_lables_np, experiment_dir, target_names=classes)

   
########################################
### INFERENCE METHODS
########################################

def inference(config_params_dict=None):
    print("run inference")

    root_path = config_params_dict['root_path']  #
    subfolder = config_params_dict['subfolder']  #
    project_name = config_params_dict['project_name']  # "StcVhhMmsiGraph"  # StcV4Graph  Cifar10Graph Caltech101Graph  StcVhhMmsiGraph
    orig_db_name = config_params_dict['orig_db_name']  # "stc_vhh_mmsi_v1_3_0"       # cifar10  stcv4  caltech101 stc_vhh_mmsi_v1_3_0
    model_architecture = config_params_dict['model_architecture']  # "sage"
    backbone = config_params_dict['backbone']  # "Resnet152"         #  Resnet152_all Resnet152_train Resnet152_val Vgg16
    subset = config_params_dict['subset']  # "train"               # train test val
    batch_size = config_params_dict['batch_size']  # 1024
    num_workers = config_params_dict['num_workers']  # 4
    k_neighbors = config_params_dict['k_neighbors']  # 1000
    sampler_sizes_l = config_params_dict['sampler_sizes_l']  # [25, 15, 5]
    hidden_channels = config_params_dict['hidden_channels']  # 256
    threshold = config_params_dict['threshold'] #30
    use_distances = config_params_dict['use_distances'] #30
    split_factor = config_params_dict['split_factor'] #30
    use_deprecated_version = config_params_dict['use_deprecated_version']
    num_heads = config_params_dict['heads']
    dropout = config_params_dict['dropout']
    aggr_mode = config_params_dict['aggr_mode']

    model_name = project_name + "_" + backbone + "_" + model_architecture
    db_name = project_name + "_" + backbone
    experiment_name = backbone + "_" + model_architecture

    db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + str(subset)
    dataset = GraphDataset(root=db_path, 
                           db_set="all", 
                           activate_masks=True, 
                           transform=None, 
                           pre_transform=None, 
                           k_neighbors=k_neighbors, 
                           threshold=threshold,
                           use_distances=use_distances,
                           split_factor=split_factor,
                           use_deprecated_version=use_deprecated_version
                           )   #NormalizeFeatures()

    test_db_path = root_path + str(backbone.lower()) + "/" + str(orig_db_name.lower()) + "/" + str(subfolder.lower()) + "/" + db_name + "_" + "test"
    test_dataset = GraphDataset(root=test_db_path, 
                           db_set="all", 
                           activate_masks=True, 
                           transform=None, 
                           pre_transform=None, 
                           k_neighbors=k_neighbors, 
                           threshold=threshold,
                           use_distances=use_distances,
                           split_factor=split_factor,
                           use_deprecated_version=use_deprecated_version
                           )   #NormalizeFeatures()
    ''''''
    print(dataset.num_node_features)
    print(dataset.num_features)
    print(dataset.num_edge_features)
    data = dataset[0]
    test_data = test_dataset[0]

    orig_features = data.x
    test_features = test_data.x
    print(orig_features.size())
    print(test_features.size())
    orig_labels = data.y
    test_labels = test_data.y
    print(orig_labels.size())
    print(test_labels.size())
    orig_attr = data.edge_attr
    test_attr = test_data.edge_attr
    print(orig_attr.size())
    print(test_attr.size())

    combined_features = torch.cat((orig_features, test_features), dim=0)
    print(combined_features.size())

    if(use_distances == False):
        combined_edges = dataset._get_adjacency_info_NEW(combined_features.detach().cpu().numpy(), return_distances=use_distances)
        print(combined_edges.size())
        combined_edge_attr = None
    else:
        combined_edges, combined_edge_attr = dataset._get_adjacency_info_NEW(combined_features.detach().cpu().numpy(), return_distances=use_distances)
        print(combined_edges.size())

    combined_labels = torch.cat((orig_labels, test_labels), dim=0)
    print(combined_labels.size())

    inference_data = Data(x=combined_features,
                          y=combined_labels,
                          edge_index=combined_edges,
                          train_mask=torch.ones((combined_features.size(0))),
                          test_mask=torch.ones((combined_features.size(0))),
                          edge_attr=combined_edge_attr
                         )
    print(inference_data)
    
    #train_idx = torch.where(data.train_mask == 1)[0]
    #test_idx = torch.where(data.test_mask == 1)[0]
    test_idx = torch.arange(inference_data.num_nodes)
    test_idx = test_idx[data.x.size(0):]   # data.x.size(0)
    #print(len(test_idx))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_loader = NeighborLoader(inference_data,
                                num_neighbors=sampler_sizes_l,
                                batch_size=batch_size,
                                input_nodes=test_idx,
                                shuffle=False,
                                num_workers=num_workers,
                                replace=True,
                                directed=True,
                                persistent_workers=False
                                )

    ''''''
    print(inference_data.num_nodes)
    print(inference_data.num_edges)
    print(inference_data.train_mask.sum())
    print(inference_data.test_mask.sum())
    
    if (model_architecture == "transformer"):
    
        model = TransformerGAT(in_channels=dataset.num_features, 
                                hidden_channels=hidden_channels, 
                                out_channels=dataset.num_classes, 
                                aggr_mode=aggr_mode,
                                dropout=dropout,
                                num_heads=num_heads,
                                return_embeddings_flag=True,
                            ).to(device)

    elif (model_architecture == "gin"):
    
        model = GraphGIN(in_channels=dataset.num_features, 
                            hidden_channels=hidden_channels, 
                            out_channels=dataset.num_classes, 
                            aggr_mode=aggr_mode,
                            dropout=dropout
                            ).to(device)

    elif(model_architecture == "sage"):
    
        model = GraphSage(in_channels=dataset.num_features, 
                          hidden_channels=hidden_channels, 
                          out_channels=dataset.num_classes, 
                          aggr_mode=aggr_mode,
                          dropout=dropout,
                          return_embeddings_flag=True,
                        ).to(device)
    
    elif(model_architecture == "gcn"):

        model = GCN(in_channels=dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    return_embeddings_flag=True,
                ).to(device)
    
    elif(model_architecture == "gat"):
    
        model = GAT(in_channels=dataset.num_features, 
                    hidden_channels=hidden_channels, 
                    out_channels=dataset.num_classes, 
                    aggr_mode=aggr_mode,
                    dropout=dropout,
                    num_heads=8,
                    return_embeddings_flag=True,
                ).to(device)
    
    else:
        print("ERROR: select valid model architecture!")
        exit()


    #wb_logger.log_model(model)

    model_states = torch.load("./" + model_name + ".pth")
    model.load_state_dict(model_states['net'])
   
    with torch.no_grad():
        total_loss = 0 
        total_correct = 0
        total_correct = 0
        all_preds = []
        all_labels = []
        all_embeddings = []
        model.eval()

        for data_batch in test_loader:
            data_batch = data_batch.to(device)
            #out, embeddings = model.batch_forward(data_batch.x, data_batch.edge_index)
            out, embeddings = model.batch_forward(data_batch)

            if embeddings != None: 
                all_embeddings.extend(embeddings[:data_batch.batch_size])
            
            pred = out.argmax(dim=-1)
            total_correct += int(out[:data_batch.batch_size].argmax(dim=-1).eq(data_batch.y[:data_batch.batch_size]).sum())
            all_preds.extend(pred[:data_batch.batch_size])
            all_labels.extend(data_batch.y[:data_batch.batch_size])

        test_acc = total_correct / test_idx.size(0)
        print(f'test accuracy: {test_acc:.4f}')   

        if embeddings != None: 
            all_embeddings_tensor = torch.stack(all_embeddings)
        all_preds_tensor = torch.stack(all_preds)
        all_labels_tensor = torch.stack(all_labels)
        if embeddings != None: 
            print(all_embeddings_tensor.size())
        print(all_preds_tensor.size())
        print(all_labels_tensor.size())

        from sklearn.metrics import classification_report, confusion_matrix
        x = all_preds_tensor.detach().cpu().numpy()
        y = all_labels_tensor.detach().cpu().numpy()
        print(classification_report(y, x))
        print(confusion_matrix(y, x))
    
    if embeddings != None: 
        print("calculate plots ... ")
        embeddings_all_np = all_embeddings_tensor.detach().cpu().numpy()
        all_lables_np = y

        print(embeddings_all_np.shape)
        print(all_lables_np.shape)

        #print("save embeddings ... ")
        #np.save("./embeddings_all_np.npy", embeddings_all_np)
        #np.save("./all_lables_np.npy", all_lables_np)

        name = model_name + "_test"
        calculate_tsne(name, embeddings_all_np, all_lables_np)
        calculate_umap(name, embeddings_all_np, all_lables_np)

def run_multiple_experiments(config_file):
    f = open(config_file, "r")
    exp_data_dict = json.load(f)
    f.close()

    for exp_data in exp_data_dict:
        config_params_dict = exp_data
        print()
        print(config_params_dict)
        train(config_params_dict=config_params_dict)

def summarize_results(results_path, subset="test"):
    print("prepare results csv")

    results_exp_list = os.listdir(results_path)
    results_exp_list = results_exp_list

    all_experiments_l = []
    for results_exp_folder in results_exp_list:
        if("2021" in results_exp_folder):
            print("Run tests on experiment folder: " + str(results_path + results_exp_folder))
            res_path = results_path + results_exp_folder
            experiment_folders_list = os.listdir(res_path)

            for experiment_folder in experiment_folders_list:
                experiment_path = os.path.join(res_path, experiment_folder)
                filelist = os.listdir(experiment_path)
                for file in filelist:
                    if( "report_" in file and subset in file and ".json" in file):
                        # load json experiment config
                        f = open(experiment_path + "/" + file, "r")
                        result_data_dict = json.load(f)
                        f.close()
                  
                        entries_l = []
                        header_l = []
                        entries_l.append(experiment_path)
                        header_l.append("experiment_path")

                        for dict_item in result_data_dict.keys():
                     
                            if(dict_item != 'accuracy' and dict_item != 'macro avg' and dict_item != 'weighted avg'):
                                precision = result_data_dict[dict_item]['precision']
                                recall = result_data_dict[dict_item]['recall']
                                f1_score = result_data_dict[dict_item]['f1-score']
                                number_of_samples = result_data_dict[dict_item]['support']

                                entries_l.extend([precision, recall, f1_score, number_of_samples])
                                header_l.extend([dict_item + "_precision", dict_item + "_recall", dict_item + "_f1_score", dict_item + "_number_of_samples"])

                            elif(dict_item == "macro avg" or dict_item == "weighted avg"):
                                precision = result_data_dict[dict_item]['precision']
                                recall = result_data_dict[dict_item]['recall']
                                f1_score = result_data_dict[dict_item]['f1-score']
                                number_of_samples = result_data_dict[dict_item]['support']

                                entries_l.extend([precision, recall, f1_score, number_of_samples])
                                header_l.extend([dict_item + "_precision", dict_item + "_recall", dict_item + "_f1_score", dict_item + "_number_of_samples"])
                            else:
                                accuracy = result_data_dict[dict_item]
                                #print(accuracy)

                                entries_l.extend([accuracy])
                                header_l.extend([dict_item])

                        #print(entries_l)
                        all_experiments_l.append(entries_l)

    header_np = np.array(header_l)
    all_experiments_np = np.array(all_experiments_l)
     
    f = open(results_path + "/results_summary.csv", 'w')
    writer = csv.writer(f, delimiter = ";")

    # write the header
    writer.writerow(header_np)

    for i in range(0, len(all_experiments_np)):
        # write the data
        writer.writerow(all_experiments_np[i])

    f.close()                        

def debug_method_1():
    np.random.seed(246)
    features_np = np.random.random((1000, 2048))
    vis_save_flag=True 
    threshold=0.07

    from sklearn import preprocessing

    v = np.array(list(features_np)).T
    adj_matrix = np.inner(v.T, v.T) / ((np.linalg.norm(v, axis=0).reshape(-1,1)) * ((np.linalg.norm(v, axis=0).reshape(-1,1)).T))
    #matrix = pd.DataFrame(sim, columns=keys, index=keys)
    print(adj_matrix.dtype)
    print(np.min(adj_matrix))
    print(np.max(adj_matrix))
    print(adj_matrix)

    min_max_scaler = preprocessing.MinMaxScaler()
    adj_matrix = min_max_scaler.fit_transform(adj_matrix)

    #New value = (value – min) / (max – min) * 100
    #adj_matrix = np.interp(adj_matrix, (adj_matrix.min(), adj_matrix.max()), (0, 1))
    #adj_matrix = (adj_matrix - np.min(adj_matrix)) / (np.max(adj_matrix) - np.min(adj_matrix))
    print(np.min(adj_matrix))
    print(np.max(adj_matrix))
    print(np.std(adj_matrix))
    print(np.mean(adj_matrix))
    print(adj_matrix)

    #adj_matrix[adj_matrix > threshold] = 1
    #adj_matrix[adj_matrix <= threshold] = 0
    print(adj_matrix)

    if(vis_save_flag == True):
        #print(np.max(adj_matrix))
        #print(np.min(adj_matrix))
        #plt.figure()
        #plt.imshow(adj_matrix, cmap='gray')
        plt.figure()
        plt.imshow(adj_matrix, cmap='gray')
        plt.figure()
        plt.imshow(adj_matrix[100:200, 100:200], cmap='gray')
        plt.show()
        #plt.savefig("./figure_" + str(features_np.shape[0]) + "_th_" + str(threshold) + ".pdf")

    row, col = np.where(adj_matrix)
    coo = np.array(list(zip(row, col)))
    distance = coo

if __name__ == "__main__":
    valid_command_list = ["train", "test", "inference", "train_saint", "train_multiple", "test_multiple", "prepare_result", "debug"]
    num_arguments = len(sys.argv) - 1

    if(num_arguments < 1 or num_arguments > 1):
        print("ERROR: specify correct commandline parameter!")
        exit()

    command = sys.argv[1]
    if command not in valid_command_list:
        print("ERROR: specify correct commandline parameter!")
        exit()

    # CONFIG for GCN
    config_params_dict =     {
        "root_path": "/caa/Projects02/vhh/private/experiments_nobackup/extracted_features/",
        "subfolder": "not_standardized",   #cropped_standardized  not_standardized   standardized
        "project_name": "StcVhhMmsiGraph",  # MovienetGraph  StcVhhMmsiGraph  HistshotDSGraph  AllMovienetGraph StcVhhMmsiGraph
        "orig_db_name": "stc_vhh_mmsi_v1_3_0",       # stc_vhh_mmsi_v1_3_0   movienet_shottypes_v2  histshotds  all_movienet_shottypes_v2  movienet_shottypes_split  
        "model_architecture": "gat",     # transformer gat  sage  gcn
        "backbone": "Resnet152",              # Resnet50  Vgg16  StcResnet50 Resnet152  MovienetResnet50
        "subset": "train",               
        "n_epochs": 500,
        "batch_size": 64,
        "num_workers": 0,
        "lr": 0.00001,
        "weight_decay": 0.0008,
        "k_neighbors": 30,
        "sampler_sizes_l": [15, 7, 2],
        "hidden_channels": [256, 128, 64],
        "early_stopping_threshold": 500,
        "lr_scheduler_patience": 250,
        "threshold": 0.25,
        "use_distances": False,
        "split_factor": 0.80,
        "use_deprecated_version": False,
        "heads": 8,
        "dropout": 0.5,
        "aggr_mode": "add"
    }

    if (command == "train"):
        train(config_params_dict=config_params_dict)
    elif (command == "train_multiple"):
        exp_list = os.listdir("./experiment_configs/")
        print(np.array(exp_list))
        exp_list = exp_list[7:8] # 3:4  6:7
        print(exp_list)
        #
        #exit()
        for exp_file in exp_list:
            print("Run experiments in json: " + str("./experiment_configs/" + exp_file))
            exp_path = "./experiment_configs/" + exp_file
            run_multiple_experiments(config_file=exp_path)

    elif (command == "test_multiple"):
        #results_path = "./results/Results_09102021/graph_classifier/"
        results_path = "./results/Results_06112021_1/"

        results_exp_list = os.listdir(results_path)
        #results_exp_list = results_exp_list[2:3]
        print(results_exp_list)
        for results_exp_folder in results_exp_list:
            if("2021" in results_exp_folder):
                print("Run tests on experiment folder: " + str(results_path + results_exp_folder))
                res_path = results_path + results_exp_folder
                experiment_folders_list = os.listdir(res_path)

                for experiment_folder in experiment_folders_list:
                    experiment_path = os.path.join(res_path, experiment_folder)
                    filelist = os.listdir(experiment_path)
                    for file in filelist:
                        if( not "report_" in file and ".json" in file):
                            # load json experiment config
                            f = open(experiment_path + "/" + file, "r")
                            exp_data_dict = json.load(f)
                            f.close()
                            print(exp_data_dict)
                            test(config_params_dict=exp_data_dict, experiment_dir=experiment_path)

    elif (command == "train_saint"):
        train_graphsaintsampler(config_params_dict=config_params_dict)
    elif (command == "test"):
        #results_path = "/caa/Homes01/dhelm/working/vhh/develop/Relation_Detection/results/20211104_MovienetGraph_movienet_shottypes_split_gat_Resnet152/exp_1/"
        results_path = "/caa/Homes01/dhelm/working/vhh/develop/Relation_Detection/results/Results_18102021_3/20211018_StcVhhMmsiGraph_stc_vhh_mmsi_v1_3_0_gat_Resnet152/exp_4/"
        test(config_params_dict=config_params_dict, experiment_dir=results_path)
    elif (command == "inference"):
        inference(config_params_dict=config_params_dict)
    elif (command == "prepare_result"):
        results_path = "./results/Results_06112021_1/"
        summarize_results(results_path)
    elif (command == "debug"):
        debug_method_1()


        