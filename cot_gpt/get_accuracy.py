import pandas as pd

lang = 'yo'

order_csv = f'outputs_resp_gpt/output_resp_{lang}.csv'
resp_csv = f'out_gpt/out_{lang}.csv'

order_df = pd.read_csv(order_csv)
resp_df = pd.read_csv(resp_csv)

ordered_resp_df = pd.merge(order_df, resp_df, 
                            on=['Start Phrase', 'Ending 1', 'Ending 2', 'Labels','Response'], 
                            how='left')

ordered_resp_df.drop('resp_json', axis=1, inplace=True)

ordered_resp_df['match'] = ordered_resp_df['Labels']==ordered_resp_df['answer']
ordered_resp_df['match'] = ordered_resp_df['match'].astype(int)

ordered_resp_df.to_csv(f'final_out_{lang}.csv',index=False)

ordered_filtered = ordered_resp_df[ordered_resp_df['match']==0]

ordered_filtered.to_csv(f'wrong_examples_gpt/wrong_{lang}.csv',index=False)

acc = ordered_resp_df['match'].mean()

acc_str = f'Accuracy on full data: {acc}'

with open(f'metrics_gpt/metric_{lang}.txt', 'w') as file:
    file.write(acc_str)
