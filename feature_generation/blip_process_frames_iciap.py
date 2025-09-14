import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from models.blip import blip_feature_extractor
from tqdm import tqdm
from PIL import Image
import os
import json


def transform_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(image)
    return image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 224
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth'
model = blip_feature_extractor(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

img_path = '../../../Agr_dataset_Collection/feature_generation/extracted_frames_v2'

path_image_features = 'blip_features_iciap/frames/'
os.makedirs(path_image_features, exist_ok=True)

museums = json.load(open('/data01/aabdari/projects/ICMR2025/museums.json', 'r'))

list_videos = list()
for m in museums:
    for r in m['rooms']:
        list_videos.extend(m['rooms'][r])

print('Num total vids', len(list_videos))
list_videos = list(set(list_videos))
print('Num unique vids', len(list_videos))

counter = 0
for v in tqdm(list_videos):
    mus_images = []
    jpg_images = [x for x in os.listdir(img_path + os.sep + v)]
    jpg_images.sort()
    torch_imgs = torch.empty(len(jpg_images), 3, 224, 224)
    for idx, img in enumerate(jpg_images):
        torch_imgs[idx, :, :, :] = transform_image(Image.open(img_path + os.sep + v + os.sep + img), image_size).unsqueeze(0).to(device)
        counter += 1
    torch_imgs = torch_imgs.to(device)
    image_features = list()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx in range(torch_imgs.shape[0]):
            image_features.append(model(image=torch_imgs[idx, :, :, :].unsqueeze(0), caption=None, mode='image', device=device)[0, 0])
        # image_features = model.encode_image(torch_imgs)
        image_features = torch.stack(image_features)
        image_features = image_features.cpu()
    torch.save(image_features, f'{path_image_features}{v}.pt')

print(counter)