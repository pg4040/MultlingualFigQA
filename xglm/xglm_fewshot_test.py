import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import XGLMTokenizer, XGLMForCausalLM

model1 = 'facebook/xglm-564M'
model2 = 'facebook/xglm-4.5B'
model3 = 'facebook/xglm-1.7B'
model4 = 'facebook/xglm-2.9B'
model5 = 'facebook/xglm-7.5B'

#####CHOSEN
model = model1
modelstr = model.split('/')[1]
tokenizer = XGLMTokenizer.from_pretrained(model)
model = XGLMForCausalLM.from_pretrained(model)


def get_logprobs(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
    return logprobs

# Zero-shot evaluation for the Choice of Plausible Alternatives (COPA) task.
# A return value of 0 indicates that the first alternative is more plausible,
# while 1 indicates that the second alternative is more plausible.
def COPA_eval(fewshot, prompt, alternative1, alternative2):
    lprob1 = get_logprobs(fewshot+"\n" + prompt + "\n" + alternative1).sum()
    lprob2 = get_logprobs(fewshot+"\n" + prompt + "\n" + alternative2).sum()
    return 0 if lprob1 > lprob2 else 1
def compute_metric(predicted, true):

    if len(predicted) != len(true):
        raise ValueError("Length of predicted and true lists must be the same.")

    accuracy = np.mean(np.array(predicted) == np.array(true))
    return accuracy
def get_fewshot_prompt(lang, K):
    example_file = f'../fewshotdata/{lang}/{lang}_{K}.csv'
    df = pd.read_csv(example_file)
    prompt = "" #ONLY ADD THE CORRECT ANSWER
    for idx, example in df.iterrows():
        ending = example['ending1'] if example['labels']=='0' else example['ending2']
        prompt+=example["startphrase"] + "\n" +  ending +"\n"
    return prompt    

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
for lang in langs_list[1:]:
    K = 10
    fewshot = get_fewshot_prompt(lang, K)

    in_csv = f'../testdata/{lang}.csv'
    in_df = pd.read_csv(in_csv)
    predicted, true = [], []
    for idx, example in in_df.iterrows():
        predict = COPA_eval(fewshot, example["startphrase"], example["ending1"], example["ending2"])
        print(f'{lang}-{idx}', predict, example['labels'])
        predicted.append(predict)
        true.append(example['labels'])
    acc = compute_metric(predicted, true)
    print(f"{K} Few shot performance for {lang} with XGLM : {acc}")
    out_df = pd.DataFrame({'startphrase':example['startphrase'],'ending1':example['ending1'],
                            'ending2':example['ending2'], 'labels':example['labels'], 
                            'predicted label': predicted})
    out_csv = f'./outputs/{modelstr}_fewshot_{K}_predictions_{lang}.csv'
    out_df.to_csv(out_csv)
# en-0 1 1
# en-1 0 0
# zh-0 1 1
# zh-1 0 0
# hi-0 1 1
# hi-1 0 0

