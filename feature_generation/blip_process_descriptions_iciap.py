import torch
from models.blip import blip_feature_extractor
import os, json
from tqdm import tqdm


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
    model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
    model.eval()
    model = model.to(device)

    path_queries_features = 'blip_features_iciap/descriptions/sentences/'
    os.makedirs(path_queries_features, exist_ok=True)

    path_descriptions = '/data01/aabdari/projects/ICMR2025/descriptions_v2/'
    list_txt_files = os.listdir(path_descriptions)

    for mus in tqdm(list_txt_files):
        file = open(path_descriptions + os.sep + mus)
        text = file.read()
        split_sentence_level = text.split('.')
        features = list()
        with torch.no_grad():
            for sent in split_sentence_level[:-1]:
                features.append(model(image=None, caption=sent, mode='text', device=device)[0, 0])
            features = torch.stack(features)
            print(features.shape)
            features = features.cpu()
            name_file = mus.replace('.txt', '.pt')
            torch.save(features, f'{path_queries_features}{name_file}')
