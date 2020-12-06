import os

from DeepPurpose.dataset import *
import DTI_inspire as models
import matplotlib.pyplot as plt
import pandas as pd
from utils_inspire import *

def main(args):
    drug_encoding, target_encoding = 'Transformer', 'CNN_inspire'

    train = pd.read_pickle('data/G35_data_train.pkl')
    val = pd.read_pickle('data/G35_data_vali.pkl')
    test = pd.read_pickle('data/G35_data_test.pkl')

    config = generate_config(drug_encoding = drug_encoding, 
                         target_encoding = target_encoding, 
                         cls_hidden_dims = [1024,1024,512],  
                         train_epoch = 100, 
                         LR = 0.0001, 
                         batch_size = 64,
                         inspire_activation = 'elu',
                         CNN_inspire_filters = 128,
                         protein_strides = [10, 15, 20, 25, 30],
                         inspire_dropout =  0,
                         protein_layers =  [128],
                         transformer_emb_size_drug = 128,
                         transformer_intermediate_size_drug = 512,
                         transformer_num_attention_heads_drug = 8,
                         transformer_n_layer_drug = 8,
                         transformer_dropout_rate = 0.1,
                         transformer_attention_probs_dropout = 0.1,
                         transformer_hidden_dropout_rate = 0.1,
                         num_workers = 8,
                         decay=0.0001
                        )
    model = models.model_initialize(**config)
    model.train(train, val, test, drop_last = True, save_path='models/second_trial_run')
    # model.save_model('models/second_trial_run')
if __name__ == '__main__':
    main(sys.argv[1:])