import pandas as pd
import numpy as np

def clean_df(out_df):
    #default 1 if not 0,1
    lt = []
    for i in range(len(out_df)):
        if out_df[i] == '0':
            lt.append(0)
        else:
            lt.append(1)
    return np.array(lt)
def compute_accuracy(in_df, out_df):
    if len(in_df) != len(out_df):
        raise ValueError("Input and output DataFrames must have the same length.")
    out_df = clean_df(out_df)
    num_matches = (in_df == out_df).sum()
    accuracy = (num_matches / len(in_df)) * 100
    return accuracy

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]


for lang in langs_list[:1]:
    model = 'vicuna-13b-v1.3'
    for K in [0, 2, 10, 20]:
        method = 'zeroshot'
        if K!=0:
            method = f'fewshot_{K}'
        in_csv = f'outputs/{model}_{lang}_{method}_predictions.csv'
        in_df = pd.read_csv(in_csv)
        acc = compute_accuracy(in_df['labels'].to_numpy(), in_df['predicted label'].to_numpy())
        print(f"{method} performance of {model} for {lang} : {acc}")
