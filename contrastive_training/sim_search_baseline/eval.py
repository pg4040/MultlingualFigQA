#For each record in the dataset, calculate 3 embeddings, and calculate cosine and euclidean similarities and get the new predictions
#Calculate per language accuracies, overall accuracy for both models cosine and euclid

import pandas as pd
import numpy as np
from sentence_embeddings import *
from collections import defaultdict

def get_combined_data():
    langs = ['en_dev','hi','jv','su','yo','id','kn','sw']
    lang_dfs = []
    for lang in langs:
        df = pd.read_csv(f'./testdata/{lang}.csv')
        df['lang'] = lang
        lang_dfs.append(df)
    return pd.concat(lang_dfs, ignore_index=True)

def get_similarity(sent, m0, m1, method='cosine'):
    if method == 'euclid':
        #compare norm of sent-m0 and sent-m1 and return which is lower
        dist_m0 = np.linalg.norm(sent-m0)
        dist_m1 = np.linalg.norm(sent-m1)
        return 0 if dist_m0 < dist_m1 else 1
    elif method == 'cosine':
        #compare normalized dot product of (sent,m0) and (sent,m1) and return which is higher
        cos_m0 = np.dot(sent, m0) / np.linalg.norm(m0) #ignoring norm(sent) as its present in both terms
        cos_m1 = np.dot(sent, m1) / np.linalg.norm(m1) #ignoring norm(sent) as its present in both terms
        return 0 if cos_m0 < cos_m1 else 1

def get_matrix_similarity(A, B, C, method='cosine'):
    if method == 'euclid':
        dist_AB = torch.sqrt(torch.sum((A - B)**2, axis=1))
        dist_AC = torch.sqrt(torch.sum((A - C)**2, axis=1))
        return torch.where(dist_AB < dist_AC, 0, 1)
    elif method == 'cosine':
        sim_AB = (A*B).sum(axis=1) / (torch.norm(A, dim=1) * torch.norm(B, dim=1))
        sim_AC = (A*C).sum(axis=1) / (torch.norm(A, dim=1) * torch.norm(C, dim=1))
        return torch.where(sim_AB > sim_AC, 0, 1).tolist()
#Individual evaluation - Batch on Row
def get_prediction_row_batched(sent, m0, m1, model, sim_method):
    sentEmb, m0Emb, m1Emb = get_sentence_embeddings([sent, m0, m1], model)
    return get_similarity(sentEmb, m0Emb, m1Emb, sim_method)
def get_prediction_row(sent, m0, m1, model, sim_method):
    sentEmb = get_sentence_embeddings([sent], model)[0]
    m0Emb = get_sentence_embeddings([m0], model)[0]
    m1Emb = get_sentence_embeddings([m1], model)[0]
    return get_similarity(sentEmb, m0Emb, m1Emb, sim_method)

def get_prediction_batched_taskwise(sents, m0s, m1s, model, sim_method):
    sentEmbs = get_sentence_embeddings(sents, model)
    m0Embs = get_sentence_embeddings(m0s, model)
    m1Embs = get_sentence_embeddings(m1s, model)
    #As,Bs,Cs - for each row in A[i], find out if it is more similar to B[i] or C[i]
    #For (1:all), we can do matrix multiplication

    return get_matrix_similarity(sentEmbs, m0Embs, m1Embs, sim_method)

def evaluate_df(df, mode, col_str, new_cols, batch_size=64):
    if mode == MODES[1]:
        batch_size = batch_size
        sents, m0s, m1s = [], [], []

    for i, row in df.iterrows():
        lang, sent, m0, m1, true = row['lang'],row['startphrase'],row['ending1'],row['ending2'],row['labels']
        if mode == MODES[0]:
            pred = get_prediction_row_batched(sent, m0, m1, model, sim_method)
            new_cols[col_str].append(pred)
        elif mode == MODES[1]:
            sents.append(sent),m0s.append(m0),m1s.append(m1)
            if i%batch_size == batch_size-1:
                preds = get_prediction_batched_taskwise(sents, m0s, m1s, model, sim_method)
                new_cols[col_str]+=preds
                sents, m0s, m1s = [], [], [] 
            print(f"Progress {i}/ {len(df)}: {i/len(df)*100}...", end='\r')
    
    if len(sents) != 0:
        preds = get_prediction_batched_taskwise(sents, m0s, m1s, model, sim_method)
        new_cols[col_str]+=preds   
    return new_cols

if __name__ == '__main__':
    # df = get_combined_data()
    df = pd.read_csv('all_langs_preds_t1_all_batched.csv')
    MODELS_LIST = ['xlm-roberta-base', 'facebook/xlm-roberta-xl'] #can also be xlm-roberta-large, xxl
    SIMS_LIST = ['cosine','euclid']
    MODES = ["sequential_row_batched", "all_batched"]
    methods_list = [(x,y) for x in MODELS_LIST for y in SIMS_LIST]
    new_cols = defaultdict(list)

    
    mode = MODES[1]
    for model, sim_method in methods_list[-1:]: 
        col_str = f'pred_{model}_{sim_method}'
        print(col_str)
        new_cols = evaluate_df(df, mode, col_str, new_cols)


    for col_str in new_cols:
        df[col_str] = new_cols[col_str]
    df.to_csv('all_langs_preds_t1_all_batched.csv', index=False) 
