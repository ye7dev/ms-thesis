import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel,  GPT2Tokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import time 
import os 
import json
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'Current device is {device}')

def parse_args():
    args = ArgumentParser()
    args.add_argument("--config", type=str)
    args.add_argument("--input_path", type=str)
    args.add_argument("--data_path", type=str, default='./../generate/results/generated_sentences.json')
    args.add_argument("--num_sent", type=int, default=1)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--lr", type=float, default=1e-04) 
    args.add_argument("--checkpoint", type=str, default='./checkpoints') 
    return args.parse_args()

def filter_sentence(trio, sentences):
    pre, post, ban = trio 
    filtered = []
    for sent in sentences:
        words = sent.split()
        if pre[0].isupper():
            ban_upper = [ban, ban[0].lower()+ban[1:]]
            overlap = set(ban_upper).intersection(words)
            if len(overlap) == 0:
                filtered.append(sent)
        else:
            if ban not in sent.split():
                filtered.append(sent)
    return filtered

def get_sentence(input_path, data_path, num_sent):
    df = pd.read_csv(input_path)
    cols = ['Preceding word','Following word','Banned word']
    trio_list = df[cols].values.tolist()
    with open(data_path) as file:
        sentence_dict = json.load(file)
    train_sentences = []
    for trio in trio_list: 
        str_key = '_'.join(trio)
        sentences = filter_sentence(trio, sentence_dict[str_key])
        if len(sentences) > num_sent:
            sentences = sentences[:num_sent]
        train_sentences.append(sentences)
    train_sentences = [sent for sent_list in train_sentences for sent in sent_list]
    return train_sentences

def get_config(path):
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    values = filename.split('_')
    cond1, cond2 = values[-2], values[-1]
    return cond1, cond2

class GPT2Dataset(Dataset):
    def __init__(self, sentences, tokenizer, batch_size):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        print("Tokenizing sentences...")
        for sent in tqdm(sentences, total=len(sentences), leave=True):
            if ')' in sent:
                continue
            if batch_size != 1:
                encoded_dict = tokenizer(sent, padding='max_length', max_length=64)
            else:
                encoded_dict = tokenizer(sent)
            self.input_ids.append(torch.tensor(encoded_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encoded_dict['attention_mask']))
        print(f'Number of train sentences: {len(self.input_ids)}')
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
    
def main(args):
    if 'eps_cutoff' in args.config:
        eps, cutoff = get_config(args.input_path)
        ckpt_path = ''
    elif 'epoch_batch' in args.config:
        ckpt_path = f'./checkpoints/{args.config}/{args.epochs}_{args.batch_size}.pth'
    else: # num_sent
        ckpt_path = f'./checkpoints/{args.config}/{args.epochs}_{args.num_sent}.pth'
        
    if os.path.exists(ckpt_path):
        print(f"{os.path.basename(ckpt_path)} already done. Finishing train function.")
        return
    
    # tokenizer, model, optimizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # dataset, dataloader
    sentences = get_sentence(args.input_path, args.data_path, args.num_sent)   
    train_dataset = GPT2Dataset(sentences, tokenizer, args.batch_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # train
    model.train()
    print('Training begins...')
    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_loss = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
            input_ids, attn_masks = batch[0], batch[1]
            labels = batch[0].clone()
            if args.batch_size != 1: # loss mask to labels
                labels[~attn_masks.bool()] = -100
            # move tensors to gpu 
            input_ids = input_ids.to(device)
            attn_masks = attn_masks.to(device)
            labels = labels.to(device)
            # forward pass
            optimizer.zero_grad()
            # output keys:['loss', 'logits', 'past_key_values']
            output = model(input_ids, attention_mask=attn_masks, labels=labels) 
            loss = output.loss
            # loss: batch average loss
            epoch_loss += loss.item()
            # backward pass
            loss.backward()
            # optimizaiton
            optimizer.step()
        end_time = time.time()
        elapsed_time = end_time - start_time
        epoch_loss /= len(train_loader)
        print(f'Epoch: {epoch}, Training loss: {epoch_loss}, Elapsed time: {elapsed_time:.2f} seconds')
    print('Training finished!')
    
    # model save path 
    new_dir = os.path.join(args.checkpoint, args.config)
    os.makedirs(new_dir, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved model to {os.path.basename(ckpt_path)}!" + '\n')
            
if __name__ == '__main__':
    args = parse_args()
    print("Continued Pretrain GPT2:")
    print(f" - config: {args.config}")
    print(f" - input_path: {args.input_path}")
    print(f" - data_path: {args.data_path}")
    print(f" - num_sentence: {args.num_sent}")
    print(f" - epochs: {args.epochs}")
    print(f" - batch_size: {args.batch_size}")
    print(f" - learning rate: {args.lr}")
    print(f" - checkpoint: {args.checkpoint}")
    print("="*20)
    start_time = time.time()
    main(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))
    print("="*20)