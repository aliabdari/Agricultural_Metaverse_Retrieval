from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
from transformers import CLIPVisionModelWithProjection
import os
import numpy as np
import torch
import json
from tqdm import tqdm


def preprocess(size, n_px):
    return Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])(n_px)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)
model = CLIPVisionModelWithProjection.from_pretrained("Searchium-ai/clip4clip-webvid150k")
model = model.eval()
model.to(device)

museums = json.load(open('../museums.json', 'r'))

list_videos = list()
for m in museums:
    for r in m['rooms']:
        list_videos.extend(m['rooms'][r])

print('Num total vids', len(list_videos))
list_videos = list(set(list_videos))
print('Num unique vids', len(list_videos))

list_fail = list()

img_path = '/data01/aabdari/projects/Agr_dataset_Collection/feature_generation/extracted_frames_v2/'

path_video_features = 'clip4clip_features/videos/'
os.makedirs(path_video_features, exist_ok=True)

counter = 0
for v in tqdm(list_videos):
    mus_images = []
    jpg_images = [x for x in os.listdir(img_path + os.sep + v)]
    jpg_images.sort()
    # numpy_imgs = np.empty(len(jpg_images), 3, 224, 224)
    numpy_imgs = list()
    for idx, img in enumerate(jpg_images):
        numpy_imgs.append((Image.open(img_path + os.sep + v + os.sep + img).convert("RGB")).resize((224, 224)))
        # numpy_imgs[idx, :, :, :] = preprocess(Image.open(img_path + os.sep + v + os.sep + img))
        counter += 1
    numpy_imgs = np.stack(numpy_imgs)
    print(numpy_imgs.shape)
    size = 224
    # try:
    with torch.no_grad():
        num_frames = numpy_imgs[1:50, :, :].shape[0] - 1
        print(num_frames)
        selected_indices = np.linspace(1, num_frames, 32, dtype=int)
        selected_frames = numpy_imgs[selected_indices, :, :]
        images = np.zeros([len(selected_frames), 3, size, size], dtype=np.float32)
        print(len(selected_frames))
        print(selected_frames.shape)
        # inputs = image_processor(list(selected_frames), return_tensors="pt")
        for i, im in enumerate(selected_frames):
            last_frame = i
            images[i, :, :, :] = preprocess(size, Image.fromarray(im).convert("RGB"))
        images = images[:last_frame + 1]
        video_frames = torch.tensor(images)
        video_frames = video_frames.to(device=device)
        # cur_video = preprocess(size, selected_frames)
        print('cur', video_frames.shape)
        visual_output = model(video_frames)
        # Normalizing the embeddings and calculating mean between all embeddings.
        visual_output = visual_output["image_embeds"]
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = torch.mean(visual_output, dim=0)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        print(visual_output.shape)
        torch.save(visual_output.cpu(), f'{path_video_features}{v}.pt')
    # except:
    #     list_fail.append(v)

print('Count Image', counter)
print('failing list', list_fail)


