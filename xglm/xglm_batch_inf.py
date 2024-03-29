import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import XGLMTokenizer, XGLMForCausalLM
import time

model1 = 'facebook/xglm-564M'
model2 = 'facebook/xglm-4.5B'
model3 = 'facebook/xglm-1.7B'
model4 = 'facebook/xglm-2.9B'
model5 = 'facebook/xglm-7.5B'

#####CHOSEN
model = model3
modelstr = model.split('/')[1]
tokenizer = XGLMTokenizer.from_pretrained(model)
model = XGLMForCausalLM.from_pretrained(model)

def get_logprobs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return logprobs


def COPA_eval_batch(prompt_list1, prompt_list2):
    lprob1 = get_logprobs(prompt_list1).unsqueeze(2).sum(dim=1)
    lprob2 = get_logprobs(prompt_list2).unsqueeze(2).sum(dim=1)
    return (lprob1 > lprob2).int().squeeze().tolist()

def compute_metric(predicted, true):

    if len(predicted) != len(true):
        raise ValueError("Length of predicted and true lists must be the same.")

    accuracy = np.mean(np.array(predicted) == np.array(true))
    return accuracy

def seq_inference():
    for lang in langs_list[:1]:
        in_csv = f'../testdata/{lang}.csv'
        in_df = pd.read_csv(in_csv)
        predicted, true = [], []
        for idx, example in in_df.iterrows():
            predict = COPA_eval(example["startphrase"], example["ending1"], example["ending2"])
            print(f'{lang}-{idx}', predict, example['labels'])
            predicted.append(predict)
            true.append(example['labels'])
        acc = compute_metric(predicted, true)
        print(f"Zero shot performance for {lang} with XGLM : {acc}")
        out_df = pd.DataFrame({'startphrase':example['startphrase'],'ending1':example['ending1'],
                                'ending2':example['ending2'], 'labels':example['labels'], 
                                'predicted label': predicted})
        out_csv = f'./outputs/{modelstr}_predictions_{lang}.csv'
        out_df.to_csv(out_csv)

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
for lang in langs_list[1:2]:
    in_csv = f'../testdata/{lang}.csv'
    print(f"INPUT FILE: {in_csv}")
    in_df = pd.read_csv(in_csv)
    prompts_list1 = []
    prompts_list2 = []
    for idx, example in in_df.iterrows():
        p1 = example['startphrase']+'\n'+example['ending1']
        p2 = example['startphrase']+'\n'+example['ending2']
        prompts_list1.append(p1)
        prompts_list2.append(p2)

    batch_size = 16
    predictions = []
    for i in range(0, len(prompts_list1), batch_size):
        sub_prompt1, sub_prompt2 = prompts_list1[i:i+batch_size], prompts_list2[i:i+batch_size]
        st = time.time()
        predict_list = COPA_eval_batch(sub_prompt1, sub_prompt2)
        en = time.time()
        predictions += predict_list
        print(f"Batch {i//batch_size}: Time: {en-st}")
    out_df = pd.DataFrame({'startphrase':example['startphrase'],'ending1':example['ending1'],
                            'ending2':example['ending2'], 'labels':example['labels'], 
                            'predicted label': predictions})
    out_csv = f'./outputs/{modelstr}_batch_predictions_{lang}.csv'
    out_df.to_csv(out_csv)
 
