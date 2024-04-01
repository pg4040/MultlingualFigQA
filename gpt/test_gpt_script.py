from openai import OpenAI

import time
import pandas as pd
api_key_file_path = '/Users/shreyat/api_keys/openai2.txt'
def read_api_key_from_file(file_path):
    with open(file_path, 'r') as file:
        api_key = file.read().strip()
    return api_key

client = OpenAI(api_key=read_api_key_from_file(api_key_file_path))


num_stories = 10
prompts = ["Once upon a time,"] * num_stories
content = "Once upon a time,"

def sequential_gpt():
    for _ in range(num_stories):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": content}],
            max_tokens=20,
        )

        # print story
        print(content + response.choices[0].message.content)


# batched example, with 10 stories completions per request
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    prompt=prompts,
    max_tokens=20,
)

# match completions to prompts by index
stories = [""] * len(prompts)
for choice in response.choices:
    stories[choice.index] = prompts[choice.index] + choice.text

# print stories
for story in stories:
    print(story)

