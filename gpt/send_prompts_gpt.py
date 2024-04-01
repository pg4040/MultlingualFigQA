from openai import OpenAI

import time
import pandas as pd
api_key_file_path = '/Users/shreyat/api_keys/openai2.txt'
def read_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

client = OpenAI(api_key=read_api_key_from_file(api_key_file_path))


new_free_engine_name = "gpt-3.5-turbo-0301"

def send_prompts(prompts):
    responses = []
    for i, prompt in enumerate(prompts):
        try:
            response = client.chat.completions.create(
                model = "gpt-3.5-turbo-0301",
                messages=[{'role':'user', 'content':prompt}],
                #prompt=prompt,
                max_tokens = 400)
            print(response.choices[0].message.content)
            responses.append(response.choices[0].message.content.strip())
            if i%3 == 2:
                time.sleep(60)  # Adjust as needed to stay within the rate limits
        except Exception as e:
            print(e)
            time.sleep(60)
            response = client.chat.completions.create(
                model = "gpt-3.5-turbo",
                messages=[{'role':'user', 'content':prompt}],
                #prompt=prompt,
                max_tokens = 400)
            print(response.choices[0].message.content)
            responses.append(response.choices[0].message.content.strip())
            
    return responses

def write_to_file(responses, csv_path):
    with open('csv_path', 'w') as csv_file:
        csv_file.write("\n".join(responses))

def gpt_per_lang(lang):
    output_csv = f'./outputs/gpt_response_{lang}_batch.csv'
    final_out_csv = f'./outputs/gpt_final_output_{lang}.csv'
    input_csv = f'./gptdata/{lang}_batch.csv'
    print(f"INPUT FILE : {input_csv}")
    prompt_df = pd.read_csv(input_csv)
    prompts = prompt_df['String']
    print(f"No. of prompts = {len(prompts)}")
    responses = send_prompts(prompts)


    df_output = pd.DataFrame({'Prompt': prompts, 'Response': responses})
    df_output.to_csv(output_csv, index=False)
    
    write_to_file(responses, final_out_csv)


def all_gpt_experiments():
    for lang in langs_list[7:]:
        gpt_per_lang(lang)
    time.sleep(60)

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
fsKs = [20]
for K in fsKs:
    prompt_list = []
    for lang in langs_list[1:]:
        in_csv = f'./gptdata/few_shot_{K}_{lang}_batch.csv'
        in_df = pd.read_csv(in_csv)
        prompt = in_df['String'].iloc[0]
        prompt_list.append(prompt)
    responses = send_prompts(prompt_list)
    
    output_csv = f'./outputs/few_shot_{K}_all_langs.csv'
    df_output = pd.DataFrame({'Prompt': prompt_list, 'Response': responses})
    df_output.to_csv(output_csv, index=False)
    
    #write_to_file(responses, final_out_csv)
     
