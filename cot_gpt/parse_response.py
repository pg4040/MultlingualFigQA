import pandas as pd
import json

lang = "hi"
df = pd.read_csv(f"output_resp_{lang}.csv")

def parse_json(data):
    try:
        return json.loads(data)
    except ValueError as e:
        return "NA"
    
df['resp_json'] = df['Response'].apply(parse_json)

def extract_answer(json_str):
    try:
        data = json.loads(json_str)
        for key, value in data.items():
            if key.lower() == 'answer':
                return value
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
    except TypeError as e:
        print(f"Invalid input: {e}")
    return None

df_filtered = df[df['resp_json'] != "NA"]

df_rem1 = df[df['resp_json'] == "NA"]

df_rem1.loc[:,'answer'] = "NA"

df_filtered.loc[:, 'answer'] = df_filtered['Response'].apply(extract_answer)

df_combined = pd.concat([df_filtered, df_rem1], axis=0)

df_combined.to_csv(f'out_gpt/out_{lang}.csv', index=False, encoding='utf-8')