from sentence_transformers import SentenceTransformer, readers, losses, InputExample, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

#####################
def modify_triplets_to_contrastive(train_examples, dev_examples, test_examples, all_examples):
    def modification(examples):
        ex = []
        for eg in examples:
            a, p, n = eg.texts
            ex.append(InputExample(texts=[a,p], label=1))
            ex.append(InputExample(texts=[a,n], label=0))
        return ex
    return modification(train_examples), modification(dev_examples), modification(test_examples), modification(all_examples)

def read_data(pref, triplets=True):
    tp_reader = readers.TripletReader('triplets_data', has_header=True, delimiter=',')
    all_examples = tp_reader.get_examples(f'{pref}.csv')
    train_examples = tp_reader.get_examples(f'{pref}_train.csv')
    dev_examples = tp_reader.get_examples(f'{pref}_dev.csv')
    test_examples = tp_reader.get_examples(f'{pref}_test.csv')
    if not triplets:
        train_examples, dev_examples, test_examples, all_examples = modify_triplets_to_contrastives(train_examples, dev_examples, test_examples, all_examples)
    print(len(train_examples), len(dev_examples), len(test_examples), len(all_examples))
    return train_examples, dev_examples, test_examples, all_examples
def get_dataloaders(train_examples, dev_examples, test_examples, all_examples, batch_size):
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size = batch_size)
    dev_dataloader = DataLoader(dev_examples, shuffle=True, batch_size = batch_size)
    test_dataloader = DataLoader(test_examples, shuffle=True, batch_size = batch_size)
    all_dataloader = DataLoader(all_examples, shuffle=True, batch_size = batch_size)
    return train_dataloader, dev_dataloader, test_dataloader, all_dataloader
def get_matrix_similarity(A, B, C, method='euclid'):
    if method == 'euclid':
        dist_AB = np.sqrt(np.sum((A - B)**2, axis=1))
        dist_AC = np.sqrt(np.sum((A - C)**2, axis=1))
        return np.where(dist_AB < dist_AC, 1, 0).tolist()
    elif method == 'cosine':
        sim_AB = np.sum(A*B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
        sim_AC = np.sum(A*C, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(C, axis=1))
        return np.where(sim_AB > sim_AC, 1, 0).tolist()

def get_preds(model, ancs, poss, negs):
    AEmbs = model.encode(ancs)
    PEmbs = model.encode(poss)
    NEmbs = model.encode(negs)
    return get_matrix_similarity(AEmbs, PEmbs, NEmbs)

def get_accuracy_datafile(filename, model, batch_size=64, dist='euclid'):
    df = pd.read_csv(filename) #This should have 'lang' also
    ancs, poss, negs, preds = [], [], [], []
    for i, row in df.iterrows():
        a,p,n = row['anchor'], row['positive'], row['negative']
        ancs.append(a), poss.append(p), negs.append(n)
        if i%batch_size == batch_size-1:
            preds += get_preds(model, ancs, poss, negs)
            ancs, poss, negs = [], [], []
    if len(ancs):
        preds += get_preds(model, ancs, poss, negs)
        ancs, poss, negs = [], [], []

    df['preds'] = preds
    total_acc = df['preds'].mean()  # Total acc

    lang_wise_acc = {}
    for lang, group in df.groupby('lang'):
        acc = group['preds'].mean()  # Accuracy for the lang
        lang_wise_acc[lang] = acc

    return total_acc, lang_wise_acc

def func(score, epoch, steps):
    print(f"Score:{score}, Epoch:{epoch}, Steps:{steps}")

####################


model = SentenceTransformer('facebook/xlm-roberta-xl')
#model = SentenceTransformer('xlm-roberta-large')
train_triplets, dev_triplets, test_triplets, all_triplets = read_data('all_langs_pos_neg_triplets', triplets=True)
batch_size = 32

#EVALUATION
tot_acc, lang_acc = get_accuracy_datafile('triplets_data/all_langs_pos_neg_triplets.csv', model)
print("Total Acc :", tot_acc)
print("Lang wise Acc :", lang_acc)

exit()
#TRAINING
train_examples, dev_examples, test_examples, all_examples = modify_triplets_to_contrastive(train_triplets, dev_triplets, test_triplets, all_triplets)
train_dl, dev_dl, test_dl, all_dl = get_dataloaders(train_examples, dev_examples, test_examples, all_examples, batch_size)
train_loss = losses.OnlineContrastiveLoss(model=model) #TO HYPERPARAMETER TUNE THE MARGIN, EUCLIDEAN DISTANCE IS THE METRIC
dev_evaluator = evaluation.TripletEvaluator.from_input_examples(dev_triplets)
model.fit(train_objectives=[(train_dl, train_loss)], epochs=10, evaluator=dev_evaluator, evaluation_steps=187, save_best_model=True, output_path='XLMR_large_model.pth', callback=func)
tot_acc, lang_acc = get_accuracy_datafile('triplets_data/all_langs_pos_neg_triplets_test.csv', model)
print("Total Acc :", tot_acc)
print("Lang wise Acc :", lang_acc)

