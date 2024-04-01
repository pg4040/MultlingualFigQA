import pandas as pd
import numpy as np

def compute_accuracy(in_df, out_df):
    if len(in_df) != len(out_df):
        raise ValueError("Input and output DataFrames must have the same length.")

    accuracy = np.mean(np.array(out_df) == np.array(in_df))
    return accuracy
    num_matches = (in_df == out_df).sum()
    accuracy = (num_matches / len(in_df)) * 100
    return accuracy

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]

for lang in langs_list[1:4]:
    model = 'xglm-564M'
    in_csv = f'outputs/{model}_predictions_{lang}.csv'
    in_df = pd.read_csv(in_csv)
    acc = compute_accuracy(in_df['labels'].to_numpy(), in_df['predicted label'].to_numpy())
    print(f"Zero shot performance of {model} for {lang} : {acc}")
