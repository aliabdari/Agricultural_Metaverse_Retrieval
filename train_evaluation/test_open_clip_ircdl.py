from DNNs import GRUNet, VideoModel, RoomModel, MuseumModel
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Subset, DataLoader
from torch.optim.lr_scheduler import StepLR
import train_utility
from Data_utils import DescriptionScene
from tqdm import tqdm
import Constants
import argparse
import numpy as np
import random


def uniform_sample(tensor, num_samples=32):
    total = tensor.shape[0]
    indices = torch.linspace(0, total - 1, steps=num_samples).long()
    return tensor[indices]


def prepare_input(output_data, list_nums):
    input_model = list()
    idx_nvr = 0
    for x in list_nums:
        input_model.append(output_data[idx_nvr:idx_nvr + x])
        idx_nvr += x
    input_model = pad_sequence(input_model, batch_first=True)
    input_model = torch.transpose(input_model, 1, 2)

    return input_model


def collate_fn(data):
    # desc
    tmp_description = [x[0] for x in data]
    tmp = pad_sequence(tmp_description, batch_first=True)
    descs = pack_padded_sequence(tmp,
                                 torch.tensor([len(x) for x in tmp_description]),
                                 batch_first=True,
                                 enforce_sorted=False)
    # museums
    tmp_mus = [x[1] for x in data]
    # videos = [y for x in tmp_mus for y in x]
    videos = [uniform_sample(z[:50, :]) for x in tmp_mus for y in x for z in y]
    frames = pad_sequence(videos, batch_first=True)
    frames = torch.transpose(frames, 1, 2)

    # num_videos_in_room = [[len(y) for y in x] for x in tmp_mus]
    num_videos_in_room = [len(y) for x in tmp_mus for y in x]
    num_rooms = [len(x) for x in tmp_mus]

    return descs, frames, num_videos_in_room, num_rooms


def start_test():
    approach_name = Constants.mode_open_clip
    mode = Constants.mode_open_clip_ircdl

    output_feature_size = 256
    batch_size = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_desc = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=True)
    model_video = VideoModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=512)
    model_room = RoomModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=512)
    model_museum = MuseumModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=256)

    model_desc.to(device)
    model_video.to(device)
    model_room.to(device)
    model_museum.to(device)

    descriptions_path, videos_path, museum_path = train_utility.get_entire_data(mode=mode)

    print(descriptions_path)
    print()

    dataset = DescriptionScene(data_description_path=descriptions_path, data_video_path=videos_path,
                               data_museum_path=museum_path, mode=mode, mem=True)

    test_indices = list(range(83))
    test_subset = Subset(dataset, test_indices)

    test_loader = DataLoader(test_subset, batch_size=64, collate_fn=collate_fn, shuffle=False, num_workers=4)

    bm_video, bm_room, bm_museum, bm_desc = train_utility.load_best_model(approach_name)
    model_video.load_state_dict(bm_video)
    model_room.load_state_dict(bm_room)
    model_museum.load_state_dict(bm_museum)
    model_desc.load_state_dict(bm_desc)

    model_video.eval()
    model_room.eval()
    model_museum.eval()
    model_desc.eval()
    output_description_test = torch.empty(len(test_indices), output_feature_size)
    output_museum_test = torch.empty(len(test_indices), output_feature_size)
    # Evaluate test set
    with torch.no_grad():
        for j, (data_desc, data_frame, data_nvr, data_r) in enumerate(test_loader):

            data_desc = data_desc.to(device)
            data_frame = data_frame.to(device)

            output_desc = model_desc(data_desc)
            output_video = model_video(data_frame)

            input_room_model = prepare_input(output_data=output_video, list_nums=data_nvr)
            output_room = model_room(input_room_model, data_nvr)

            input_museum_model = prepare_input(output_data=output_room, list_nums=data_r)
            output_museum = model_museum(input_museum_model, data_r)

            initial_index = j * batch_size
            final_index = (j + 1) * batch_size
            if final_index > len(test_indices):
                final_index = len(test_indices)
            output_description_test[initial_index:final_index, :] = output_desc
            output_museum_test[initial_index:final_index, :] = output_museum
    ds1, ds5, ds10, sd1, sd5, sd10, ndgc_10, ndcg, ds_medr, sd_medr = train_utility.evaluate(
        output_description=output_description_test,
        output_scene=output_museum_test,
        section="test")


if __name__ == '__main__':
    start_test()
