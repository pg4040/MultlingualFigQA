from collections import defaultdict
import pandas as pd
import numpy as np

def compute_accuracy(in_df, out_df):
    if len(in_df) != len(out_df):
        raise ValueError("Input and output DataFrames must have the same length.")

    num_matches = (in_df['labels'] == out_df['predicted label']).sum()
    accuracy = (num_matches / len(in_df)) * 100
    return accuracy
def compute_accuracy(in_df, out_df):
    if len(in_df) != len(out_df):
        raise ValueError("Input and output DataFrames must have the same length.")

    num_matches = (in_df == out_df).sum()
    accuracy = (num_matches / len(in_df)) * 100
    return accuracy

def lang_metric(lang):
    in_csv = f'./testdata/{lang}.csv'
    out_csv = f'./outputs/gpt_final_output_{lang}.csv'
    in_df = pd.read_csv(in_csv)
    out_df = pd.read_csv(out_csv, header=None, names=['index', 'predicted label'])
    accuracy = compute_accuracy(in_df, out_df)
    print(f"Zero shot performance for {lang} : {accuracy}")

def per_lang_metric(K=2):
    batch_size_map = defaultdict(lambda:20)
    batch_size_map['hi'] = 5
    batch_size_map['kn'] = 5
    batch_size_map['yo'] = 5
    comb_csv = f'./outputs/few_shot_{K}_all_langs.csv'
    comb_df = pd.read_csv(comb_csv)
    in_csv_list = []
    
    for i, lang in enumerate(langs_list[1:]):
        try:
            in_csv = f'./testdata/{lang}.csv'
            in_df = pd.read_csv(in_csv)
            batch_size = batch_size_map[lang]
            labels = in_df['labels'].tolist()[:batch_size] 
            lang_response = comb_df.iloc[i]['Response']
            pred_list = [int(line.split(',')[-1].strip()) for line in lang_response.split('\n')[1:]]
            print(labels)
            print(pred_list) 
            lang_acc = compute_accuracy(np.array(pred_list), np.array(labels))
            print(f"Few shot {K} {lang} accuracy: {lang_acc}")
        except:
            print("Output formmating issue for {lang}")
langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
def metric_compute(): 
    lang = 'en_dev'
    for lang in langs_list:
        lang_metric(lang)
per_lang_metric(20) 
