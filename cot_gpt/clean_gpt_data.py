import pandas as pd

def generate_instruction(row):
    return f"Instruction: Each question is a metaphor, and the options represent two possible meanings. Select the meaning which is closest to the metaphor. Explain the reasoning and output the option index as the answer in the json format: Explanation:... Answer:... Where Startphrase: {row['startphrase']}, Options: 0) {row['ending1']} 1) {row['ending2']}"
    
def clean_data(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)
    df['instruction_column'] = df.apply(generate_instruction, axis=1)
    df.to_csv(output_csv_path, index=False)
    

languages = ["hi", "id", "jv", "kn", "su", "sw", "yo"]

for lang in languages:
    input_csv_path = f'test_data/{lang}.csv'
    output_csv_path = f'cleaned_gpt_data/{lang}.csv'
    clean_data(input_csv_path, output_csv_path)