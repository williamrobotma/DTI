""" File: G35_predict.py
Developer(s): William, Ma
Date: <date-you-submit>
---
Description: This is a prediction script that
encodes new Drug-Protein pairs and predicts a
continuous log-transformed binding affinity.
"""
import pandas as pd
import numpy as np
import pickle as pkl
# from DeepPurpose.utils import *
# TODO: Include any other librariies you might need. 
#       Comment the version used (e.g. use 'pip freeze' command)
import DTI_inspire as models # It is important that this is imported AFTER DeepPurpose
from utils_inspire import * # It is important that this is imported AFTER DeepPurpose
from torch.utils.data import SequentialSampler
import torch
# Python version 3.7.9
# pytorch=1.4.0, torchvision=0.5.0


def _load_model(model_path='G35_model'):
    """ _load_model
       Helper function that loads the trained model, unpickles it,
       and returns it to generate predictions.
       ---
       Input: <str> filename, the path to the trained model.
       Output: <model obj>, the deserialized model
    """
    # TODO: Change the default filename to your group id
    model = models.model_pretrained(path_dir=model_path)
    return model 

def encode_pairs(df_pairs):
    """ encode_pairs
        An input dataframe of drug SMILES and protein sequences
        is encoded into the numerical representation required by
        the <G#_model.pkl> model to generate predictions.
        ---
        Input: <pd.DataFrame> df_pairs, a dataframe with three columns,
               the first with heading "SMILES" containing the string
               representation of a drug molecule, the second with 
               heading "Target Sequence" containing the string 
               representation of a protein target, and  a third
               column "Label" with the float values of binding
               affinities.
        Output: <pd.DataFrame> df_encode, a dataframe with five columns; 
               the same "SMILES", "Target Sequence", & "Labels" in addition to 
               the encoded representations: "drug_encoding" and 
               "target_encoding".
    """

    # drug_encoding, target_encoding = 'Transformer', 'CNN_inspire'
    # train, val, test  = data_process(df_pairs['SMILES'].to_numpy(), df_pairs['Target Sequence'].to_numpy(), df_pairs['Label'], 
    #                   drug_encoding, target_encoding, 
    #                   split_method='cold_protein',frac=[1.0, 0.0, 0.0],
    #                   cnn_inspire_use_transformer_embedding=False, # new embedding set
    #                   random_seed = 42)

    # assert len(train) == len(df_pairs)

    # train = train.rename(columns={'Label':'Labels'})
    df_encode = df_pairs.copy()
    drug_encoding, target_encoding = 'Transformer', 'CNN_inspire'
    df_encode = encode_drug(df_encode, drug_encoding)
    df_encode = encode_protein(df_encode, target_encoding, cnn_inspire_use_transformer_embedding=False)

    return df_encode


def predict_pairs(df_encode, batch_size=1):
    """ predict_pairs
        For all pairs in a dataframe containing drug and target encodings,
        the pretrained model is used to generate a predicted binding afinity
        (float value) that is added as a column to the dataframe named "predicted".
        ---
        Input: <pd.DataFrame> df_encode, a five colunm dataframe with the columns
              "drug_encoding" and "target_encoding" that are processed and passed
              the trained model to generate predicted score.
        Output: <pd.DataFrame> df_results, a six column dataframe with the same five
              columns and a sixth named "predicted" containing the float value
              generated by the model for each input drug-target pair.
    """
    df_results = df_encode.copy()
    model = _load_model()
    # TODO: Implement any required processing to generate predictions with your model.
    #       For example, concatenate the representations if that is required by your
    #       model. Then generate a score for each pair in the dataframe (e.g. a loop).
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model.model.to(device)
    model.model.eval()


    info = data_process_loader(df_results.index.values, df_results.Label.values, df_results, **model.config)

    params = {'batch_size': batch_size,
            'shuffle': False,
            'num_workers': 0,
            'drop_last': False,
            'sampler':SequentialSampler(info)}


    generator = data.DataLoader(info, **params)

    y_pred = []
    # y_label = []

    for i, (v_d, v_p, label) in enumerate(generator):

        # print(v_d)
        # print(v_p)
        # print(label)
        v_d = v_d           

        v_p = v_p.float().to(device)                
        score = model.model(v_d, v_p)

        logits = torch.squeeze(score).detach().cpu().numpy()
        # label_ids = torch.from_numpy(np.asarray(label)).to('cpu').numpy()
        # label_ids = label.to('cpu').numpy()
        # y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    df_results['predicted'] = y_pred
    # df_results['new_label'] = y_label


    return df_results


# FIN: There shouldnt be anything outside of the methods, nor a main function.
#      Implement any additional methods as "helper" function, denoted by a 
#      leading underscore (e.g. _load_model).
