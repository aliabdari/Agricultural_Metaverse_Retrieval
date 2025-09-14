'''
This script is developed to extract descriptions features using MobileClip model
'''
import torch
import open_clip
import os
from tqdm import tqdm
import pickle
from nltk.tokenize import WordPunctTokenizer


def tokenize_paragraph_with_punctuations(paragraph):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(paragraph)
    return tokens


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:apple/MobileCLIP-S1-OpenCLIP')
tokenizer = open_clip.get_tokenizer('hf-hub:apple/MobileCLIP-S1-OpenCLIP')
model.eval()
model.to(device=device)


path_descriptions = '/data01/aabdari/projects/ICMR2025/descriptions_v2/'
list_txt_files = os.listdir(path_descriptions)

path_sentence_level_features = '/data01/aabdari/projects/ICMR2025/feature_generation/mobile_clip_features/descriptions/sentences/'

os.makedirs(path_sentence_level_features, exist_ok=True)

for mus in tqdm(list_txt_files):
    file = open(path_descriptions + os.sep + mus)
    text = file.read()
    split_sentence_level = text.split('.')
    split_sentence_level_tokenized = tokenizer(split_sentence_level[:-1])
    split_sentence_level_tokenized = split_sentence_level_tokenized.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        features_sentences = model.encode_text(split_sentence_level_tokenized)

        file_name = mus.replace('.txt', '')
        features_sentences = features_sentences.cpu()
        torch.save(features_sentences, f'{path_sentence_level_features}{file_name}.pt')
