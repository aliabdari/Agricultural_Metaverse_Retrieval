'''
This script is developed to extract descriptions features using openclip model
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
model_name = 'ViT-B-32'
pre_train_dataset = 'laion2b_s34b_b79k'
model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pre_train_dataset)
model.eval()
model.to(device=device)
tokenizer = open_clip.get_tokenizer(model_name)


path_descriptions = '../descriptions_ircdl/'
list_txt_files = os.listdir(path_descriptions)

path_sentence_level_features = '../feature_generation/open_clip_features_ircdl/descriptions/sentences/'
path_token_level_features = '../feature_generation/open_clip_features_ircdl/descriptions/tokens/'
path_token_strings = '../feature_generation/open_clip_features_ircdl/descriptions/tokens_strings/'

os.makedirs(path_sentence_level_features, exist_ok=True)
os.makedirs(path_token_level_features, exist_ok=True)
os.makedirs(path_token_strings, exist_ok=True)

for mus in tqdm(list_txt_files):
    file = open(path_descriptions + os.sep + mus)
    text = file.read()
    split_sentence_level = text.split('.')
    split_sentence_level_tokenized = tokenizer(split_sentence_level[:-1])
    split_sentence_level_tokenized = split_sentence_level_tokenized.to(device)

    split_token_level = tokenize_paragraph_with_punctuations(text)
    split_token_level_tokenized = tokenizer(split_token_level)
    split_token_level_tokenized = split_token_level_tokenized.to(device)
    if len(split_token_level_tokenized) != len(split_token_level):
        print(len(split_token_level_tokenized))
        print(len(split_token_level))
        exit(0)
    with torch.no_grad(), torch.cuda.amp.autocast():
        features_sentences = model.encode_text(split_sentence_level_tokenized)
        features_tokens = model.encode_text(split_token_level_tokenized)

        file_name = mus.replace('.txt', '')
        features_sentences = features_sentences.cpu()
        torch.save(features_sentences, f'{path_sentence_level_features}{file_name}.pt')

        features_tokens = features_tokens.cpu()
        torch.save(features_tokens, f'{path_token_level_features}{file_name}.pt')
        with open(f'{path_token_strings}{file_name}.pkl', 'wb') as f:
            pickle.dump(split_token_level, f)
