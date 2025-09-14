'''
This script is developed to extract descriptions features using openclip model
'''
import torch
import open_clip
import os
from tqdm import tqdm
import pickle
from nltk.tokenize import WordPunctTokenizer
import numpy as np
import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection


def tokenize_paragraph_with_punctuations(paragraph):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(paragraph)
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = CLIPTextModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
tokenizer = CLIPTokenizer.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model.eval()
model.to(device=device)


path_descriptions = '/data01/aabdari/projects/ICMR2025/descriptions_v2/'
list_txt_files = os.listdir(path_descriptions)

path_sentence_level_features = '/data01/aabdari/projects/ICMR2025/feature_generation/clip4clip_features/descriptions/sentences/'

os.makedirs(path_sentence_level_features, exist_ok=True)

for mus in tqdm(list_txt_files):
    file = open(path_descriptions + os.sep + mus)
    text = file.read()
    split_sentence_level = text.split('.')
    split_sentence_level = split_sentence_level[:-1]
    # split_sentence_level_tokenized = tokenizer(split_sentence_level[:-1], return_tensors="pt")
    # split_sentence_level_tokenized = split_sentence_level_tokenized.to(device)
    embeddings = list()
    with torch.no_grad():
        for sent in split_sentence_level:
            inputs = tokenizer(sent, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            final_output = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
            final_output = final_output.cpu().detach()
            embeddings.append(torch.squeeze(final_output))
        embeddings = torch.stack(embeddings)
        print('shape embeddings', embeddings.shape)
        file_name = mus.replace('.txt', '')
        torch.save(embeddings, f'{path_sentence_level_features}{file_name}.pt')