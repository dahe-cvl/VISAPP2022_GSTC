import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models, datasets
import torch.nn as nn  
from torch.autograd import Variable
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv
import json
import os
import sys
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from Utils import calculate_tsne, calculate_umap, plot_confusion_matrix
from Logger import Logger


def loadDatasetFromFolder(path="", batch_size=64, split_factor=0.8):
    """
    This method is used to load a specified dataset.
    :param path: [required] path to dataset folder holding the subfolders "train", "val" and "test".
    :param batch_size: [optional] specifies the batchsize used during training process.
    :return: instance of trainloader, validloader, testloader as well as the corresponding dataset sizes
    """

    if (path == "" or path == None):
        print("ERROR: you must specifiy a valid dataset path!")
        exit()

    # Datasets from folders
    traindir = path + "/train/"  # train_nara_loc_efa  train_nara train  train_paper  test
    testdir = path + "/test/"  #test_user_study   test  test_paper_loc_est  test_cinescale_tiny

    # Number of subprocesses to use for data loading
    num_workers = 4

    # Convert data to a normalized torch.FloatTensor
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # randomly flip and rotate
        transforms.RandomVerticalFlip(),  # randomly flip and rotate
        transforms.RandomRotation(10),
        transforms.ToTensor(),
       
        transforms.Normalize([75.67079594559294 / 255.0,
                                68.76940725676867 / 255.0,
                                62.73719133427122 / 255.0],
                             [66.50369750799024 / 255.0,
                                64.23437522274287 / 255.0,
                                62.36074514298541 / 255.0
                            ])
    ])

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

    '''
    transforms.Normalize([75.67079594559294 / 255.0,
                            68.76940725676867 / 255.0,
                            62.73719133427122 / 255.0],
                            [66.50369750799024 / 255.0,
                            64.23437522274287 / 255.0,
                            62.36074514298541 / 255.0
                        ])
    '''

    train_data = datasets.ImageFolder(root=traindir, transform=transform_train)
    test_data = datasets.ImageFolder(root=testdir, transform=transform_test)

    dataset_size = len(train_data)
    indices = list(range(dataset_size))
    split = int(np.floor(split_factor * dataset_size))
    np.random.seed(12)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    trainloader = DataLoader(train_data, 
                            batch_size=batch_size, 
                            sampler=train_sampler,
                            num_workers=num_workers)

    validloader = DataLoader(train_data, 
                            batch_size=batch_size,
                            sampler=valid_sampler,
                            num_workers=num_workers)

    testloader = DataLoader(test_data,
                            batch_size=batch_size,
                            num_workers=num_workers
                            )

    print("train samples: " + str(len(train_indices)))
    print("valid samples: " + str(len(val_indices)))
    print("test samples: " + str(len(test_data)))

    return trainloader, len(train_indices), validloader, len(val_indices), testloader

def loadModel(model_arch="", classes=None, pre_trained_path=None):
    """
    This module is used to load specified deep learning model.
    :param model_arch: string value [required] - is used to select between various deep learning architectures
     (Resnet, Vgg, Densenet, Alexnet)
    :param classes: list of strings [required] - is used to hold the class names (e.g. ['ELS', 'LS', 'MS', 'CU'])
    :param pre_trained_path: string [optional] - is used to specify the path to a pre-trained model
    :return: the specified instance of the model
    """

    print("Load model architecture ... ")
    if (model_arch == "resnet50"):
        print("Resnet architecture selected ...")

        model = models.resnet50(pretrained=True)
        for params in model.parameters():
            params.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))

        if (pre_trained_path != None):
            print("load pre_trained weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net']) #, strict=False

    elif (model_arch == "resnet18"):
        print("Resnet architecture selected ...")

        model = models.resnet18(pretrained=True)

        for params in model.parameters():
            params.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))

        if (pre_trained_path != None):
            print("load pre_trained weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net']) #, strict=False

    elif (model_arch == "resnet152"):
        print("Resnet architecture selected ...")

        model = models.resnet152(pretrained=True)

        for params in model.parameters():
            params.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, len(classes))

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))

        if (pre_trained_path != None):
            print("load pre_trained weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net']) #, strict=False

    elif (model_arch == "vgg16"):
        print("Vgg architecture selected ...")

        model = models.vgg16(pretrained=True)
        # print(model)

        for params in model.parameters():
            params.requires_grad = True

        layers = model.children()
        print("number of layers: " + str(type(layers)))

        for params in model.parameters():
            params.requires_grad = True

        model.classifier[-1] = torch.nn.Linear(4096, len(classes))
        # print(model)
        # exit()
        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))

        if (pre_trained_path != None):
            print("load pre_trained weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net']) #, strict=False

    elif (model_arch == "densenet121"):
        print("Densenet architecture selected ...")

        model = models.densenet121(pretrained=True)
        # print(model)

        for params in model.parameters():
            params.requires_grad = True

        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Linear(num_ftrs, len(classes))
        # print(model)

        # Find total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print("total_params:" + str(total_params))
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print("total_trainable_params: " + str(total_trainable_params))

        if (pre_trained_path != None):
            print("load pre_trained weights ... ")
            model_dict_state = torch.load(pre_trained_path)
            model.load_state_dict(model_dict_state['net']) #, strict=False
    else:
        model = None
        print("ERROR: select valid model architecture!")
        exit()

    return model

########################################
### TRAINING METHODS
########################################

def train(config_params_dict):
    print("run training")

    root_path = config_params_dict['root_path']  #
    subfolder = config_params_dict['subfolder']  #
    project_name = config_params_dict['project_name']  # "StcVhhMmsiGraph"  # StcV4Graph  Cifar10Graph Caltech101Graph  StcVhhMmsiGraph
    orig_db_name = config_params_dict['orig_db_name']  # "stc_vhh_mmsi_v1_3_0"       # cifar10  stcv4  caltech101 stc_vhh_mmsi_v1_3_0
    model_architecture = config_params_dict['model_architecture']  # "cnn"
    backbone = config_params_dict['backbone']                    #  resnet50 vgg16
    subset = config_params_dict['subset']  # "train"               # train test val
    n_epochs = config_params_dict['n_epochs']  # 1000
    batch_size = config_params_dict['batch_size']  # 1024
    num_workers = config_params_dict['num_workers']  # 4
    lRate = config_params_dict['lr']  # 0.001
    wDecay = config_params_dict['weight_decay']  # 0.08
    early_stopping_threshold = config_params_dict['early_stopping_threshold'] #60
    lr_scheduler_patience = config_params_dict['lr_scheduler_patience'] #30
    split_factor = config_params_dict['split_factor'] #30
    dropout = config_params_dict['dropout']
    classes = config_params_dict['classes']
    pre_trained_weights = None

    model_name = project_name + "_" + backbone + "_" + model_architecture
    db_name = project_name + "_" + backbone
    experiment_name = backbone + "_" + model_architecture
    db_path = root_path

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

    wb_logger = Logger()
    
    wb_logger.set_general_experiment_params(
        exp_config=config_params_dict,
        experiment_dir="./cnn_relations/",
        project_name=project_name,
        experiment_name=experiment_name
    )

    ################
    # load dataset
    ################
    trainloader, nSamples_train, validloader, nSamples_valid, testloader = loadDatasetFromFolder(db_path, batch_size, split_factor=split_factor)

    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # define model
    ################
    model = loadModel(model_arch=backbone, classes=classes, pre_trained_path=pre_trained_weights)
    #if(pre_trained_path != None):
    #    model_dict_state = torch.load(pre_trained_path)
        #model.load_state_dict(model_dict_state['net'])

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    ################
    # Specify the Loss function
    ################
    criterion = nn.CrossEntropyLoss()

    ################
    # Specify the optimizer
    ################
    optimizer = optim.SGD(model.parameters(), lr=lRate, momentum=0.9, nesterov=True, weight_decay=wDecay)

    print("[Creating Learning rate scheduler...]")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=0.00001, patience=lr_scheduler_patience, verbose=True)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    wb_logger.initialize_logger()
    wb_logger.log_model(model)

    # Define the lists to store the results of loss and accuracy
    best_acc = 0.0
    best_loss = 10.0
    early_stopping_cnt = 0

    for epoch in range(0, n_epochs):
        tLoss_sum = 0
        tAcc_sum = 0
        vLoss_sum = 0
        vAcc_sum = 0
        ###################
        # train the model #
        ###################
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # run forward pass
            outputs = model(inputs)
            tLoss = criterion(outputs, labels)
            tLoss_sum += tLoss.item()

            # run backward pass
            optimizer.zero_grad()
            tLoss.backward()
            optimizer.step()

            preds = outputs.argmax(1, keepdim=True)
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / preds.shape[0]
            tAcc_sum += acc.item()

        ###################
        # validate the model #
        ###################
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(validloader):
                # If we have GPU, shift the data to GPU
                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                vLoss = criterion(outputs, labels)
                vLoss_sum += vLoss.item()

                preds = outputs.argmax(1, keepdim=True)
                correct = preds.eq(labels.view_as(preds)).sum()
                acc = correct.float() / preds.shape[0]
                vAcc_sum += acc.item()

        train_loss = tLoss_sum / len(trainloader)
        train_acc = tAcc_sum / len(trainloader)
        val_loss = vLoss_sum / len(validloader)
        val_acc = vAcc_sum / len(validloader)

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
        acc_curr = 100. * (vAcc_sum / len(validloader))
        vloss_curr = vLoss_sum / len(validloader)
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
            torch.save(state, experiment_dir + "/" + model_name + ".pth")  # + str(round(acc_curr, 4)) + "_" + str(round(vloss_curr, 4))
            best_acc = acc_curr
            # best_loss = vloss_curr
            early_stopping_cnt = 0
        
        scheduler.step(val_loss)

        ###############################
        # early stopping.
        ###############################
        if (acc_curr <= best_acc):
            early_stopping_cnt = early_stopping_cnt + 1
        if (early_stopping_cnt >= early_stopping_threshold):
            print('Early stopping active --> stop training ...')
            break

    ###################
    # validate the model #
    ###################
    # final evaluation on test set 
    model_states = torch.load(experiment_dir + "/" + model_name + ".pth")
    model.load_state_dict(model_states['net'])

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        vAcc_sum = 0
        all_preds = []
        all_labels = []
        all_embeddings = []
        for i, (inputs, labels) in enumerate(testloader):
            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            embeddings = None

            if embeddings != None: 
                all_embeddings.extend(embeddings)

            preds = outputs.argmax(1, keepdim=True)
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / preds.shape[0]
            vAcc_sum += acc.item()

            all_preds.extend(preds)
            all_labels.extend(labels)

        val_acc = vAcc_sum / len(testloader)
        print(f'test accuracy: {val_acc:.4f}')   

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
        #classes = ["CU", "ELS", "I", "LS", "MS", "NA"]
        classes = ["CS", "ECS", "FS", "LS", "MS"]
        x = all_preds_tensor.detach().cpu().numpy()
        y = all_labels_tensor.detach().cpu().numpy()

        report_dict = classification_report(y, x, target_names=classes, output_dict=True)
        matrix = confusion_matrix(y, x)

        with open(experiment_dir + "/report_test_" + model_name + ".json", 'w') as fp:
            json.dump(report_dict, fp)

        # save confusion matrix 
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=True,
                              path=experiment_dir + "/test_confusion_matrix_normalize.png")
        plot_confusion_matrix(cm=matrix, target_names=classes, title='Confusion matrix', cmap=None, normalize=False,
                              path=experiment_dir + "/test_confusion_matrix.png")

    if embeddings != None: 
        print("calculate tsne and umap plots ... ")
        embeddings_all_np = all_embeddings_tensor.detach().cpu().numpy()
        all_lables_np = y

        print(embeddings_all_np.shape)
        print(all_lables_np.shape)

        name = model_name + "_test"
        calculate_tsne(name, embeddings_all_np, all_lables_np, experiment_dir)
        calculate_umap(name, embeddings_all_np, all_lables_np, experiment_dir)

    wb_logger.finish_run()


#########################
# test section
#########################

def test(config_params_dict, experiment_dir="./"):
    print("run test")

    root_path = config_params_dict['root_path']  #
    subfolder = config_params_dict['subfolder']  #
    project_name = config_params_dict['project_name']  # "StcVhhMmsiGraph"  # StcV4Graph  Cifar10Graph Caltech101Graph  StcVhhMmsiGraph
    orig_db_name = config_params_dict['orig_db_name']  # "stc_vhh_mmsi_v1_3_0"       # cifar10  stcv4  caltech101 stc_vhh_mmsi_v1_3_0
    model_architecture = config_params_dict['model_architecture']  # "cnn"
    backbone = config_params_dict['backbone']                    #  resnet50 vgg16
    subset = config_params_dict['subset']  # "train"               # train test val
    batch_size = config_params_dict['batch_size']  # 1024
    num_workers = config_params_dict['num_workers']  # 4
    classes = config_params_dict['classes']
    pre_trained_weights = None

    model_name = project_name + "_" + backbone + "_" + model_architecture
    db_name = project_name + "_" + backbone
    experiment_name = backbone + "_" + model_architecture
    db_path = root_path
    
    ################
    # load dataset
    ################
    trainloader, nSamples_train, validloader, nSamples_valid, testloader = loadDatasetFromFolder(db_path, batch_size)

    # Whether to train on a gpu
    train_on_gpu = torch.cuda.is_available()
    print("Train on gpu: " + str(train_on_gpu))

    # Number of gpus
    multi_gpu = False
    if train_on_gpu:
        gpu_count = torch.cuda.device_count()
        print("gpu_count: " + str(gpu_count))
        if gpu_count > 1:
            multi_gpu = True
        else:
            multi_gpu = False

    ################
    # define model
    ################
    model = loadModel(model_arch=backbone, classes=classes, pre_trained_path=pre_trained_weights)
    model_dict_state = torch.load(experiment_dir + "/" + model_name + ".pth")
    model.load_state_dict(model_dict_state['net'])

    if train_on_gpu:
        model = model.to('cuda')

    if multi_gpu:
        model = nn.DataParallel(model)

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_embeddings = []
        vAcc_sum = 0
        for i, (inputs, labels) in enumerate(testloader):
            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            embeddings = None

            if embeddings != None: 
                all_embeddings.extend(embeddings)

            preds = outputs.argmax(1, keepdim=True)
            correct = preds.eq(labels.view_as(preds)).sum()
            acc = correct.float() / preds.shape[0]
            vAcc_sum += acc.item()

            all_preds.extend(preds)
            all_labels.extend(labels)

        val_acc = vAcc_sum / len(testloader)
        print(f'test accuracy: {val_acc:.4f}')   

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
        #classes = ["CU", "ELS", "I", "LS", "MS", "NA"]
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
        calculate_tsne(name, embeddings_all_np, all_lables_np, experiment_dir)
        calculate_umap(name, embeddings_all_np, all_lables_np, experiment_dir)

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

if __name__ == "__main__":
    valid_command_list = ["train", "test", "inference", "test_multiple", "prepare_result"]
    num_arguments = len(sys.argv) - 1

    if(num_arguments < 1 or num_arguments > 1):
        print("ERROR: specify correct commandline parameter!")
        exit()

    command = sys.argv[1]
    if command not in valid_command_list:
        print("ERROR: specify correct commandline parameter!")
        exit()

    config_params_dict = {
        #"root_path": "./stc_vhh_mmsi_1_3_0/stc_vhh_mmsi_v1_3_0/",
        "root_path": "./public_datasets/movienet/movienet_shottypes_split/",
        "subfolder": "standardized",  
        "project_name": "MovienetGraph",  
        "orig_db_name": "movienet_shottypes_split",      
        "model_architecture": "cnn",    
        "backbone": "resnet50",        
        "subset": "",               # train test val
        "n_epochs": 50,
        "batch_size": 64,
        "num_workers": 0,
        "lr": 0.0001,
        "weight_decay": 0.005,
        "early_stopping_threshold": 25,
        "lr_scheduler_patience": 15,
        "split_factor": 0.3,
        "dropout": 0.4,
        #"classes": ["CU","ELS","I","LS","MS","NA"]
        "classes": ["CS", "ECS", "FS", "LS", "MS"]
    } 

    if (command == "train"):
        train(config_params_dict=config_params_dict)
    elif (command == "test"):
        experiment_path = "./20211104_MovienetGraph_movienet_shottypes_split_cnn_resnet50/exp_1/"  
        #experiment_path = "./traditional_cnn/20211011_StcVhhMmsiGraph_stc_vhh_mmsi_v1_3_0_cnn_resnet50/exp_1/"
        test(config_params_dict=config_params_dict, experiment_dir=experiment_path)
    elif (command == "test_multiple"):
        results_path = "./results/Results_09102021/traditional_cnn/"
        results_exp_list = os.listdir(results_path)
      
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
    elif (command == "prepare_result"):
        results_path = "./results/visapp2022/"
        summarize_results(results_path)
