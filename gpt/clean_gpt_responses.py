import pandas as pd

def clean_response(response):
    pred_str_list = []
    response = response.strip().split('\n')
    idx_list = []
    for line in response:
        if ',' in line:
            idx, val = line.split(',')
            idx = idx.split(' ')[-1]
            pred_str = idx.strip()+','+val.strip()
            pred_str_list.append(pred_str)
            idx_list.append(int(idx))
        elif ':' in line:
            idx, val = line.split(':')
            idx = idx.split(' ')[-1]
            pred_str = idx.strip()+','+val.strip()
            pred_str_list.append(pred_str)
            idx_list.append(int(idx))
        else:
            print(f"Didn't consider this line: {line}")
            continue
        
    return pred_str_list, idx_list

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]
for lang in langs_list[2:]:
    in_csv = f'./outputs/gpt_response_{lang}_batch.csv'
    out_csv = f'./outputs/gpt_final_output_{lang}.csv'
    in_df = pd.read_csv(in_csv)
    responses = in_df['Response'].tolist()
    pred_str_list, idx_list = [], []
    for response in responses:
        pred_str_list.extend(clean_response(response)[0])
        idx_list.extend(clean_response(response)[1])

    metric_df = pd.read_csv(f'./testdata/{lang}.csv')
    print(lang)
    print(len(metric_df), len(pred_str_list))
    if len(metric_df) != len(pred_str_list):
        pred_idx_list = set(idx_list)
        true_idx_list = set(list(range(len(metric_df))))
        missing = true_idx_list - pred_idx_list

    with open(out_csv, 'w') as out_file:
        out_file.write('\n'.join(pred_str_list))

