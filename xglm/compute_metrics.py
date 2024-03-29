import pandas as pd
import numpy as np

def compute_accuracy(in_df, out_df):
    if len(in_df) != len(out_df):
        raise ValueError("Input and output DataFrames must have the same length.")

    num_matches = (in_df == out_df).sum()
    accuracy = (num_matches / len(in_df)) * 100
    return accuracy

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]

for lang in langs_list[2:]:
    model = 'xglm-1.7B'
    in_csv = f'outputs/{model}_batch_predictions_{lang}.csv'
    in_df = pd.read_csv(in_csv)
    acc = compute_accuracy(in_df['labels'], in_df['predicted label'])
    print(f"Zero shot performance of {model} for {lang} : {acc}")
