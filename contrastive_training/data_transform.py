import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('all_langs_preds_t1.csv')
df2 = pd.DataFrame(columns=['anchor', 'positive', 'negative', 'lang'])
df2['anchor'] = df['startphrase']
positives = []
negatives = []
for i, row in df.iterrows():
    pos = row['ending1'] if row['labels'] == 0 else row['ending2']
    neg = row['ending2'] if row['labels'] == 0 else row['ending1']
    positives.append(pos)
    negatives.append(neg)
df2['positive'] = positives
df2['negative'] = negatives
df2['lang'] = df['lang']


pref = 'all_langs_pos_neg_triplets'
df2.to_csv('all_langs_pos_neg_triplets.csv', index=False)
class_dist = df2['lang'].value_counts(normalize=True)
train_dev, test = train_test_split(df2, test_size=0.5, stratify = df2['lang'])
train, dev = train_test_split(train_dev, test_size=0.2, stratify = train_dev['lang'])
train.to_csv(f"{pref}_train.csv", index=False)
dev.to_csv(f"{pref}_dev.csv", index=False)
test.to_csv(f"{pref}_test.csv", index=False)

