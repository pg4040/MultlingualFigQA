#ZERO SHOT
import pandas as pd
from collections import defaultdict
from transformers import GPT2Tokenizer

# Initialize GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
def count_tokens(prompt):
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    return len(tokens)

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
lang = langs_list[1]

#Batch 60 examples together based on approx 3000 tokens
#En - 60, Hi - 15, Id - 60, Jv - 60, Kn - 10, Su - 60, Sw = 60, Yo - 20
batch_size_map = defaultdict(lambda:20)
batch_size_map['hi'] = 5
batch_size_map['kn'] = 5
batch_size_map['yo'] = 5

K=20
exp_name, FEW_SHOT_PROMPT = '', ''
PROMPT_FORMAT = "Question {}: {}\nOptions:\n0) {}\n1) {}\n"
if K!=0:
    exp_name = f'few_shot_{K}'
    filename = f'./fewshotdata/{lang}/{lang}_{K}.csv'
    fs_df = pd.read_csv(filename)
    prompt = ''
    label_str = 'Output:\n'
    for idx, row in fs_df.iterrows():
        question, op1, op2, label = row['startphrase'], row['ending1'], row['ending2'], row['labels']
        prompt += PROMPT_FORMAT.format(idx, question, op1, op2)
        label_str += f'{idx}, {label}\n'
    FEW_SHOT_PROMPT = f'Examples:\n{prompt}\n{label_str}\n INPUT:\n'

INSTRUCTION_CMD = f"Instruction: For each of the below questions, only output the option index as the answer. Return the answers as a csv <Question No., Option Index>\n{FEW_SHOT_PROMPT}"
def compute_gpt_prompts(lang):
    csv_file = f'./testdata/{lang}.csv'

    df = pd.read_csv(csv_file)
    prompt_list = []
    token_count = []
    gt_labels = []
    for index, row in df.iterrows():
        if len(row) != 4:
            print("Invalid row format:", row)
            continue
        question, op1, op2, label = row['startphrase'], row['ending1'], row['ending2'], row['labels']
        prompt = PROMPT_FORMAT.format(index, question, op1, op2)
        prompt_list.append(prompt)
        token_count.append(count_tokens(prompt))

    num_examples = len(prompt_list) #1000-ish
    avg_token_count = sum(token_count)/ num_examples # 35 - En, 161- Hi,42 - Id, 55-Jv, 239-Kn, 60-Su, Sw- 50, Yo - 116 

    print(f"CSV FILE: {csv_file}\n Num examples: {num_examples}\n Token count: {avg_token_count}")

    batch_size = batch_size_map[lang]
    batch_prompts_list = []
    for i in range(0, len(prompt_list), batch_size):
        subset = prompt_list[i:i+batch_size]
        batch_prompt = INSTRUCTION_CMD + '\n'.join(subset)
        batch_prompts_list.append(batch_prompt)

    batch_prompts_output_csv = f'./gptdata/{exp_name}_{lang}_batch.csv'
    pd.DataFrame({"Index": range(len(batch_prompts_list)), "String": batch_prompts_list}).to_csv(batch_prompts_output_csv, index=False)

for lang in langs_list[1:]:
    compute_gpt_prompts(lang)

