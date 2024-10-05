import torch 
from argparse import ArgumentParser
import pickle
import spacy
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd 
import numpy as np
from tqdm import tqdm
import time 

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Current device is {device}')

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.to(device)

nlp = spacy.load("en_core_web_sm")

def parse_args():
    # todo: help description
    args = ArgumentParser()
    args.add_argument("--input_file", type=str, default='supplementary/preprocess/results/automated_prompt.csv')
    args.add_argument("--epsilon", type=float)
    args.add_argument("--proportion_cut_off", type=float)
    args.add_argument("--output_dir", type=str)
    return args.parse_args()

def get_next_token_dist(p):
    input_ids = tokenizer.encode(p, return_tensors='pt').to(device) # batch_size: 1
    outputs = model(input_ids) # output shape: [logit, batch_size, seq_length, num_vocab]
    logit = outputs[0][0, -1, :] 
    next_token_prob = torch.softmax(logit, dim=-1)
    # calculate the adaptive cutoff
    eta = get_eta(args.epsilon, logit)
    # indices of tokens that satisfy the condition
    above_tokens = (next_token_prob > eta).nonzero().squeeze(1)
    above_probs = next_token_prob[above_tokens]
    # sort by probability 
    sorting_indices = torch.argsort(above_probs, descending=True)
    sorted_above_tokens = above_tokens[sorting_indices]
    sorted_above_probs = above_probs[sorting_indices]
    # save token and its probability in a dict 
    token_prob_dict = {} 
    probability_sum = sum(sorted_above_probs).item()
    for i in range(len(sorted_above_tokens)):
        # tensor to int(index, key)/float(prob, value) -> normalize 
        token_prob_dict[sorted_above_tokens[i].item()] = sorted_above_probs[i].item()/probability_sum
    return token_prob_dict

def get_eta(eps, next_token_logit):
    '''
    reference: https://github.com/huggingface/transformers/blob/1e9da2b0a6ef964c2cf72dd715dbee991a3f49fa/src/transformers/generation/logits_process.py#L399
    '''
    epsilon = torch.tensor(eps)
    entropy = torch.distributions.Categorical(logits=next_token_logit).entropy()
    eta = torch.min(epsilon, torch.sqrt(epsilon) * torch.exp(-entropy))
    return eta

def compare_gender(cut_off, male_next_dict, female_next_dict, target_pos):
    male_oriented_tokens, female_oriented_tokens =[], []
    # compare overlapped token 
    overlapped_keys = [k for k in male_next_dict.keys() & female_next_dict.keys()] # k : int
    for key in overlapped_keys:
        # int -> str -> remove whitespace
        new_key = tokenizer.decode(key).strip() 
        if len(new_key) < 3: continue
        # filter out words with non-matching tag
        docu = nlp(new_key)
        pos = [w.pos_ for w in docu]
        if target_pos not in pos: continue
        # compare probability of the token in each context 
        male_next_prob, female_next_prob = male_next_dict[key], female_next_dict[key]
        if male_next_prob > female_next_prob and (male_next_prob / female_next_prob) > cut_off: # male-oriented
            male_oriented_tokens.append(new_key)
        elif female_next_prob > male_next_prob and (female_next_prob / male_next_prob) > cut_off: # female-oriented
            female_oriented_tokens.append(new_key)
    return male_oriented_tokens, female_oriented_tokens

def get_biased_token(row):
    male_prompt, female_prompt = row['male_prompt'], row['female_prompt']
    male_noun, female_noun = row['male_noun'], row['female_noun']
    target_pos = row['target_pos']
    next_to_male= get_next_token_dist(male_prompt)
    next_to_female = get_next_token_dist(female_prompt)
    male_oriented_tokens, female_oriented_tokens = compare_gender(args.proportion_cut_off, next_to_male, next_to_female, target_pos)
    
    # fill out count info
    count_info = [{'male_prompt': male_prompt, 'female_prompt': female_prompt, \
                # number of tokens survived after eta sampling 
                'num_next_male': len(next_to_male),'num_next_female': len(next_to_female),\
                # number of biased tokens
                'num_male_biased': len(male_oriented_tokens), 'num_female_biased': len(female_oriented_tokens)}]
    
    # make triplet 
    row_list = [] 
    for i in range(len(male_oriented_tokens)):
        row_dict = {'Preceding word': female_noun,\
                    'Following word': male_oriented_tokens[i],\
                    'Banned word': male_noun,\
                    'Prompt': male_prompt}
        row_list.append(row_dict)                           
    for i in range(len(female_oriented_tokens)):
        row_dict = {'Preceding word': male_noun,\
                    'Following word': female_oriented_tokens[i],\
                    'Banned word': female_noun,\
                    'Prompt': female_prompt}
        row_list.append(row_dict)   
                                 
    return pd.DataFrame(count_info), pd.DataFrame(row_list)

def main(args):
    df = pd.read_csv(args.input_file)
    triplet_col_names = ['Preceding word', 'Following word', 'Banned word', 'Prompt']
    triplet_df = pd.DataFrame(columns=triplet_col_names)
    count_col_names = ['male_prompt','female_prompt', 'num_next_male', 'num_next_female', 'num_male_biased' , 'num_female_biased']
    count_df = pd.DataFrame(columns=count_col_names)
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        count_info, new_triplet = get_biased_token(row)  
        triplet_df = pd.concat([triplet_df, new_triplet])
        count_df = pd.concat([count_df, count_info])
        
    output_file_path = f'{args.output_dir}/biased_token/biased_tokens_{args.epsilon}_{args.proportion_cut_off}.csv'
    triplet_df.to_csv(output_file_path, index=False)
    print(f'Saved the results at {output_file_path}')
    count_file_path = f'{args.output_dir}/count_info/count_info_{args.epsilon}_{args.proportion_cut_off}.csv'
    count_df.to_csv(count_file_path, index=False)
          
if __name__ == '__main__':
    args = parse_args()
    print("Getting biased tokens:")
    print(f" - input_file: {args.input_file}")
    print(f" - epsilon: {args.epsilon}")
    print(f" - proportion_cut_off: {args.proportion_cut_off}")
    print(f" - output_directory: {args.output_dir}")
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    print("="*20)