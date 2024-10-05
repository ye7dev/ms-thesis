from argparse import ArgumentParser
import pandas as pd 
import numpy as np
from tqdm import tqdm
import openai
import time
from openai.error import RateLimitError
import backoff
import os 
import json

api_key = ""
openai.api_key = api_key 

def parse_args():
    args = ArgumentParser()
    args.add_argument("--input_dir", type=str, default='supplementary/bias-detect/results/biased_token')
    args.add_argument("--output_file", type=str, default='./results/generated_sentences.json') 
    return args.parse_args()

def merge_dfs(directory):
    df_list = []
    cols = ['Preceding word','Following word','Banned word']
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            df_list.append(df)
    concat_df = pd.concat(df_list, ignore_index=True)
    concat_df = concat_df[cols]
    print(f'Total number of triplets with duplicates: {len(concat_df)}')
    concat_df.to_csv('./results/concat_triplet.csv')
    concat_df = concat_df.drop_duplicates()
    print(f'Total number of triplets without duplicates: {len(concat_df)}')
    return concat_df

@backoff.on_exception(backoff.expo, RateLimitError)
def call_api(generation_prompt):
    for i in range(0,5):
        try:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            top_p=1.0,
            max_tokens=64,
            n=5,
            messages=[{"role": "user", "content": generation_prompt}]
            )
            sents = [completion.choices[i].message.content for i in range(5)]
            return sents
        except openai.error.APIError as e:
            print(f'API error -> {i}th retrial')
    
def main(args):   
    concat_df = merge_dfs(args.input_dir)
    trio_list = concat_df.values.tolist()
    sentence_dict = {}
    template = " ".join([
                "Make a sentence according to the instructions.", 
                "1) You must not contain the word '{banned}' in the sentence.", 
                "2) You must contain '{preceding}' and '{following}' in the sentence.",
                "3) The word '{preceding}' always precedes the word '{following}'."])  
    
    for trio in tqdm(trio_list, total=len(trio_list), leave=True):
        pre,post,ban = trio
        prompt = template.format(banned=ban, preceding=pre, following=post)
        str_key = '_'.join(trio) # to save dict to json file
        sentence_dict[str_key] = call_api(prompt)
    
    with open(args.output_file, 'w') as file:
        json.dump(sentence_dict, file)

if __name__ == '__main__':
    args = parse_args()
    print("Generate the dataset:")
    print(f" - input_dir: {args.input_dir}")
    print(f" - output_file: {args.output_file}")
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    print("="*20)


