import pandas as pd
import numpy as np

# HI
def mismatches(lang='hi'):
    print(lang)
    in_csv = f'./testdata/{lang}.csv'
    out_csv = f'./outputs/gpt_final_output_{lang}.csv'
    in_df = pd.read_csv(in_csv)
    out_df = pd.read_csv(out_csv, header=None, names=['index', 'predicted label'])
    mismatch_index = in_df[in_df['labels'] != out_df['predicted label']].index
    TPT = 'Question: {}\n Correct:{}\n Predicted:{}'
    print(len(mismatch_index))
    sample_indexes = np.random.choice(mismatch_index, size=3, replace=False)

    for index in sample_indexes:
        row = in_df.loc[index]
        correct = row['ending1'] if row['labels']==0 else row['ending2']
        pred = row['ending1'] if row['labels']==1 else row['ending2']
        print(TPT.format(row['startphrase'], correct, pred))
langs_list = ['hi', 'kn', 'id', 'sw']
for lang in langs_list:
    print(mismatches(lang))
