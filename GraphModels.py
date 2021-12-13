import torch
import torch.nn.functional as F
from torchvision import models
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import Linear


########################################
### MODELS
########################################

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, aggr_mode="mean", dropout=0.4, return_embeddings_flag=False, use_edge_attr=False):
        super(GCN, self).__init__()

        self.num_layers = len(hidden_channels)
        self.return_embeddings_flag = return_embeddings_flag
        self.use_edge_attr = use_edge_attr


        self.aggr_mode = aggr_mode
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels[0], aggr=self.aggr_mode))  
        for i in range(1, len(hidden_channels)):
            self.convs.append(GCNConv(hidden_channels[i-1], hidden_channels[i], aggr=self.aggr_mode))
        self.convs.append(GCNConv(hidden_channels[-1], out_channels, aggr=self.aggr_mode))
        #self.fc_final = Linear(hidden_channels[-1], out_channels, bias=True)

        '''
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(BatchNorm(hidden_channels[0]))
        for i in range(1, len(hidden_channels) - 1 ):
            self.batch_norms.append(BatchNorm(hidden_channels[i]))
        self.batch_norms.append(BatchNorm(hidden_channels[-1]))
        '''

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()
        #self.fc_final.reset_parameters()

    def batch_forward(self, data_batch):
        if(self.use_edge_attr ==True):
            #trans = NormalizeFeatures("edge_attr")
            #data_batch = trans(data_batch)
            edge_weight = data_batch.edge_attr.squeeze()
        else:
            edge_weight = None
        edge_index = data_batch.edge_index
        x = data_batch.x

        embeddings = None
        number_of_layer = len(self.convs) - 1

        for i in range(0, len(self.convs)):
            x = self.convs[i](x, edge_index, edge_weight)

            if(self.return_embeddings_flag == True and i == number_of_layer):
                embeddings = x

            #x = self.batch_norms[i](x)
            if(i < len(self.convs) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            #print(x.size())

        '''
        if(self.return_embeddings_flag == True):
            embeddings = x
            x = self.fc_final(x)
        else:
            x = self.fc_final(x)
        '''

        return x, embeddings

class GraphSage(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, aggr_mode="mean", dropout=0.4, return_embeddings_flag=False):
        super(GraphSage, self).__init__()

        self.num_layers = len(hidden_channels)
        self.return_embeddings_flag = return_embeddings_flag

        self.aggr_mode = aggr_mode
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels[0], aggr=self.aggr_mode))  
        for i in range(1, len(hidden_channels)):
            self.convs.append(SAGEConv(hidden_channels[i-1], hidden_channels[i], aggr=self.aggr_mode))
        self.convs.append(SAGEConv(hidden_channels[-1], out_channels, aggr=self.aggr_mode))
        
        #self.fc1 = Linear(hidden_channels[-1], 2048, bias=True)
        #self.fc2 = Linear(2048, 2048, bias=True)
        #self.fc_final = Linear(2048, out_channels, bias=True)
        #self.fc_final = Linear(hidden_channels[-1], out_channels, bias=True)

        '''
        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(BatchNorm(hidden_channels[0]))
        for i in range(1, len(hidden_channels) - 1 ):
            self.batch_norms.append(BatchNorm(hidden_channels[i]))
        self.batch_norms.append(BatchNorm(hidden_channels[-1]))
        '''

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        #self.fc_final.reset_parameters()
        
        #self.fc1.reset_parameters()
        #self.fc2.reset_parameters()

    def batch_forward(self, data_batch):
        edge_index = data_batch.edge_index
        x = data_batch.x

        embeddings = None
        number_of_layer = len(self.convs) - 2

        for i in range(0, len(self.convs)):
            x = self.convs[i](x, edge_index)
            #x = self.batch_norms[i](x)

            if(self.return_embeddings_flag == True and i == number_of_layer):
                embeddings = x

            if(i < len(self.convs) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            #print(x.size())  
             
        #x = F.relu(self.fc1(x))
        #x = F.dropout(x, p=self.dropout, training=self.training)
        #x =  F.relu(self.fc2(x)) 
        #x = F.dropout(x, p=self.dropout, training=self.training)        
        #x = self.fc_final(x)   
        '''
        if(self.return_embeddings_flag == True):
            embeddings = x
            x = self.fc_final(x)
        else:
            x = self.fc_final(x)
        '''
        return x, embeddings

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads=1, aggr_mode="mean", dropout=0.4, return_embeddings_flag=False, use_edge_attr=False):
        super(GAT, self).__init__()

        self.num_layers = len(hidden_channels)
        self.return_embeddings_flag = return_embeddings_flag
        self.use_edge_attr = use_edge_attr

        self.aggr_mode = aggr_mode
        self.dropout = 0.5

        dropout_gat = dropout
        concat_flag = True

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels[0], heads=num_heads, dropout=dropout_gat, aggr=self.aggr_mode))
        for i in range(1, len(hidden_channels)):
            self.convs.append(GATConv(hidden_channels[i-1] * num_heads, hidden_channels[i], heads=num_heads, dropout=dropout_gat, aggr=self.aggr_mode)) 
        self.convs.append(GATConv(hidden_channels[-1] * num_heads, out_channels, aggr=self.aggr_mode))
       
        '''
        if (concat_flag == True):
            self.fc_final = Linear(hidden_channels[-1] * num_heads, out_channels, bias=True)
        else:
            self.fc_final = Linear(hidden_channels[-1], out_channels, bias=True)
        '''        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        #self.fc_final.reset_parameters()

    def batch_forward(self, data_batch):
        if(self.use_edge_attr ==True):
            edge_attr = data_batch.edge_attr
        else:
            edge_attr = None

        edge_index = data_batch.edge_index
        x = data_batch.x

        embeddings = None
        number_of_layer = len(self.convs) - 2

        for i in range(0, len(self.convs)):
            x = self.convs[i](x, edge_index, edge_attr)            
            if(self.return_embeddings_flag == True and i == number_of_layer):
                embeddings = x

            if(i < len(self.convs) - 1):
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        '''
        if(self.return_embeddings_flag == True):
            embeddings = x
            x = self.fc_final(x)
        else:
            x = self.fc_final(x)
        #print(x.size())
        '''
        return x, embeddings