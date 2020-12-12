import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn 

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle 
torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

import os

from DeepPurpose.utils import *
from DeepPurpose.model_helper import Encoder_MultipleLayers, Embeddings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InspireEncoder(nn.Sequential):

    # def PLayer(self, size, filters, activation, initializer, regularizer_param):
    #     def f(input):
    #         # model_p = Convolution1D(filters=filters, kernel_size=size, padding='valid', activity_regularizer=l2(regularizer_param), kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
    #         model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
    #         model_p = BatchNormalization()(model_p)
    #         model_p = Activation(activation)(model_p)
    #         return GlobalMaxPooling1D()(model_p)
    #     return f

    def __init__(self, **config):
        super(InspireEncoder, self).__init__()

        if config['cnn_inspire_use_transformer_embedding']:
            self.emb = nn.Embedding(config['input_dim_protein'], 20)
        else:
            self.emb = nn.Embedding(26, 20)

        torch.nn.init.xavier_uniform_(self.emb.weight)

        self.embedding_dropout = nn.Dropout2d(0.2)

        self.convs = nn.ModuleList()
        for stride_size in config['protein_strides']:
            if config['inspire_activation'] == 'sigmoid':
                activation = nn.Sigmoid()
            elif config['inspire_activation'] == 'relu':
                activation = nn.ReLU()
            else:
                activation = nn.ELU()
            network = nn.Sequential(
                nn.Conv1d(in_channels=20, out_channels=config['CNN_inspire_filters'], kernel_size=stride_size, padding=stride_size // 2),
                nn.BatchNorm1d(config['CNN_inspire_filters'], eps=1e-5, momentum=0.99),
                activation,
            )
            torch.nn.init.xavier_uniform_(network[0].weight)

            self.convs.append(network)

        if config['protein_layers']:
            self.fc = nn.ModuleList()

            n_layers= len(config['protein_layers'])
            input_layer = config['CNN_inspire_filters'] * len(config['protein_strides'])
            dims = [input_layer] + config['protein_layers']
            for i in range(n_layers):
                if config['inspire_activation'] == 'sigmoid':
                    activation = nn.Sigmoid()
                elif config['inspire_activation'] == 'relu':
                    activation = nn.ReLU()
                else:
                    activation = nn.ELU()
                network = nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.BatchNorm1d(dims[i+1], eps=0.001, momentum=0.99),
                    activation,
                    nn.Dropout(config['inspire_dropout'])
                )
                torch.nn.init.xavier_uniform_(network[0].weight)
                self.fc.append(network)
        else: self.fc = None
        

        

    def forward(self, feature):
        # print(feature.shape)
        feature = self.emb(feature.long().to(device))
        # print(feature.shape)
        feature = self.embedding_dropout(feature)
        # print(feature.shape)
        feature = torch.transpose(feature, 1, 2)
        # print(feature.shape)
        feature = [torch.max(conv(feature), dim=2)[0] for conv in self.convs]
        # print(len(feature), feature[0].shape)
        # print(feature)
        feature = torch.cat(feature, axis=1)
        # print(feature.shape)
        if self.fc:
            for i, l in enumerate(self.fc):
                feature = l(feature)
        # print(feature.shape)
        return feature
    
    