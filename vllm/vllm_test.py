from vllm import LLM, SamplingParams
import pandas as pd

def get_fs_prompt(lang, K):
    if K == 0:
        return ''
    fs_csv = f'../fewshotdata/{lang}/{lang}_{K}.csv'
    fs_df = pd.read_csv(fs_csv)
    FEW_SHOT = ''
    SINGLE_LINE = 'Metaphor: {} Meaning 0: {} Meaning 1: {} Output: {}\n'
    for i, row in fs_df.iterrows():
        FEW_SHOT+=SINGLE_LINE.format(row['startphrase'], row['ending1'], row['ending2'], row['labels'])
    return FEW_SHOT

langs_list = ["en_dev", "hi", "id", "jv", "kn", "su", "sw", "yo"]

for lang in langs_list[:1]:
    in_csv = f'../testdata/{lang}.csv'
    in_df = pd.read_csv(in_csv)

    INSTRUCTION = 'Choose the meaning which is closest to the metaphor. Output 0 or 1.\n{}Metaphor: {} Meaning 0: {} Meaning 1: {}\nOutput:'
    model_name = "lmsys/vicuna-13b-v1.3"
    gpu_count = 1
    sampling_params = SamplingParams(temperature=0.7, top_p=1)
    llm = LLM(model_name, tensor_parallel_size=gpu_count)

    for K in [0, 2, 10, 20]:
        fs_prompt = get_fs_prompt(lang, K)
            
        prompt_list = []
        for i, row in in_df.iterrows():
            prompt = INSTRUCTION.format(fs_prompt, row['startphrase'], row['ending1'], row['ending2'])
            prompt_list.append(prompt)
        outputs = llm.generate(prompt_list, sampling_params)
        predictions = []
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text.strip()
            predictions.append(generated_text)
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


        in_df['predicted label'] = predictions
        modelstr = model_name.split('/')[1]
        method = 'zeroshot'
        if K!=0:
            method = f'fewshot_{K}'
        in_df.to_csv(f'outputs/{modelstr}_{lang}_{method}_predictions.csv')

