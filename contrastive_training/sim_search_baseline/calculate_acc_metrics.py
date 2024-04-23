#Given a csv file with lang, pred, true columns, overall and per language accuracy metrics
import pandas as pd

mode, model, sim_method = 'all_batched', 'facebook/xlm-roberta-xl', 'euclid'
# mode, model, sim_method = 'sequential_row_batched', 'facebook/xlm-roberta-xl', 'cosine'
# mode, model, sim_method = 'sequential_row_batched', 'xlm-roberta-base', 'euclid'
filename = f'all_langs_preds_t1_{mode}.csv'
pred_label = f'pred_{model}_{sim_method}'
print(pred_label)
df = pd.read_csv(filename)
df[pred_label] = df[pred_label].map(lambda k: int(k.strip()[-2]))
overall_accuracy = (df[pred_label] == df['labels']).mean()

# Calculate accuracy per language
accuracy_per_language = df.groupby('lang').apply(lambda x: (x[pred_label] == x['labels']).mean())

print("Overall Accuracy:", overall_accuracy)
print("\nAccuracy Per Language:")
print(accuracy_per_language)

# MODEL : XLM-R-BASE; EMBEDDING EUCLID SIMILARITY - BATCHED and PADDED.
#Overall Accuracy: 0.5305630026809651
# Accuracy Per Language:
# lang
# en_dev    0.521024
# hi        0.519000
# id        0.516667
# jv        0.503333
# kn        0.528381
# su        0.686667
# sw        0.519126
# yo        0.497260

# MODEL : XLM-R-BASE; EMBEDDING EUCLID SIMILARITY - SEQUENTIAL and NO PADDING. 
#Overall Accuracy: 0.5304289544235925

# Accuracy Per Language:
# lang
# en_dev    0.521024
# hi        0.519000
# id        0.516667
# jv        0.503333
# kn        0.527546
# su        0.686667
# sw        0.519126
# yo        0.497260
#dtype: float64
#THERE FORE BATCHING AND PADDING DOESN'T EFFECT IT TOO MUCH ====================

# XLM-R-BASE; EMBEDDING COSINE SIMILARITY - BATCHED and PADDED 
#Overall Accuracy: 0.5298927613941019

# Accuracy Per Language:
# lang
# en_dev    0.519196
# hi        0.519000
# id        0.515789
# jv        0.503333
# kn        0.528381
# su        0.686667
# sw        0.518215
# yo        0.495890
# dtype: float64
## EUCLID SLIGHTLY BETTER THAN COSINE, BUT MOSTLY SIMILAR ONLY

#ERROR: Some weights of XLMRobertaXLModel were not initialized from the model checkpoint at facebook/xlm-roberta-xl and are newly initialized: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
# xlmrobertaXL with EUCLID SIMILARITY
#Overall Accuracy: 0.5414209115281501

# Accuracy Per Language:
# lang
# en_dev    0.542048
# hi        0.499000
# id        0.562281
# jv        0.520000
# kn        0.516694
# su        0.690000
# sw        0.536430
# yo        0.509589
# dtype: float64