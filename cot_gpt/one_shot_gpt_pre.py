import pandas as pd
import time
import openai

# Reference: ChatGPT4

def read_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

api_key_file_path = 'openai_key3.txt'
api_key = read_api_key_from_file(api_key_file_path)
print("Using API Key:", api_key)
client = openai.OpenAI(api_key=api_key)


def predict_and_explain(input_csv_path, output_resp, progress_path):
    try:
        df = pd.read_csv(input_csv_path)
        last_index = 0
        try:
            with open(progress_path, 'r') as file:
                last_index = int(file.read().strip())
        except FileNotFoundError:
            last_index = 0 

        resp = []
        request_count = 0
        for index, row in df.iloc[last_index:].iterrows():
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": row['instruction_column']}],
                max_tokens=150
            )
            text = response.choices[0].message.content.strip()
            resp.append({
                "Start Phrase": row['startphrase'],
                "Ending 1": row['ending1'],
                "Ending 2": row['ending2'],
                "Labels": row['labels'],
                "Response": text
            })

            request_count += 1
            if request_count % 1000 == 0:
                time.sleep(3)
            
            with open(progress_path, 'w') as file:
                file.write(str(index))
        
        resp_df = pd.DataFrame(resp)
        resp_df.to_csv(output_resp, index=False, encoding='utf-8')

    except Exception as e:
        print(f"Error occurred: {e}")
        
languages = ["hi", "id", "jv", "kn", "su", "sw", "yo"]
#languages = ["hi"]
for lang in languages:
    input_csv_path = f'cleaned_gpt_data/{lang}.csv'
    output_resp = f'outputs_resp_gpt/output_resp_{lang}.csv'
    progress_path = f'progress_gpt/progress_{lang}.txt'
    predict_and_explain(input_csv_path, output_resp, progress_path)
