import json
import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import Constants


class DescriptionScene(Dataset):
    def __init__(self, data_description_path, data_video_path, data_museum_path, mode, mem=True):
        self.description_path = data_description_path
        self.data_frames_path = data_video_path
        self.data_museum_path = data_museum_path
        self.videos = dict()
        self.mode = mode
        if self.mode == Constants.mode_open_clip_ircdl:
            available_data = open('data/available_data_v1.txt', 'r')
        else:
            available_data = open('data/available_data.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        if self.mem:
            print('Data Loading ...')

            print('Loading descriptions ...')
            if os.path.exists(f'data/descs_{self.mode}.pkl'):
                pickle_file = open(f'data/descs_{self.mode}.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + 'Museum_' + s + '.pt'))
                pickle_file = open(f'data/descs_{self.mode}.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading Videos ...')
            self.museums = json.load(open(self.data_museum_path, 'r'))
            if os.path.exists(f'data/videos_tensors_{self.mode}.pth'):
                self.videos = torch.load(f"data/videos_tensors_{self.mode}.pth")
            else:
                list_vids = list()
                for m in self.museums:
                    for r in m['rooms']:
                        list_vids.extend(m['rooms'][r])
                list_vids = list(set(list_vids))
                self.videos = {v: torch.load(self.data_frames_path + os.sep + v + '.pt') for v in list_vids}

                torch.save(self.videos, f"data/videos_tensors_{self.mode}.pth")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            list_vids_museum = list()
            for r in self.museums[index]['rooms']:
                list_vids_museum.append([self.videos[self.museums[index]['rooms'][r][x]] for x in range(len(self.museums[index]['rooms'][r]))])
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            list_vids_museum = list()
            for r in self.museums[index]['rooms']:
                list_vids_museum.append([torch.load(self.data_frames_path + os.sep + self.museums[index]['rooms'][r][x] + '.pt')
                                         for x in range(len(self.museums[index]['rooms'][r]))])

        return desc_tensor, list_vids_museum


class DescriptionSceneCombined(Dataset):
    def __init__(self, data_description_path, data_video_path_img, data_video_path_vid, data_museum_path, mode, mem=True):
        self.description_path = data_description_path
        self.data_frames_path_img = data_video_path_img
        self.data_frames_path_vid = data_video_path_vid
        self.data_museum_path = data_museum_path
        self.videos_img = dict()
        self.videos_vid = dict()
        available_data = open('data/available_data.txt', 'r')
        self.samples = [x[:-1] for x in available_data.readlines()]
        self.mem = mem
        self.mode = mode
        if self.mem:
            print('Data Loading ...')
            if self.mode == Constants.mode_open_clip_vivit:
                desc_mod = Constants.mode_open_clip
            else:
                print('ERROR')
                exit(0)
            print('Loading descriptions ...')
            if os.path.exists(f'data/descs_{desc_mod}.pkl'):
                pickle_file = open(f'data/descs_{desc_mod}.pkl', 'rb')
                self.descs = pickle.load(pickle_file)
                pickle_file.close()
            else:
                self.descs = []
                for idx, s in tqdm(enumerate(self.samples), total=len(self.samples)):
                    self.descs.append(torch.load(self.description_path + os.sep + 'Museum_' + s + '.pt'))
                pickle_file = open(f'data/descs_{desc_mod}.pkl', 'wb')
                pickle.dump(self.descs, pickle_file)
                pickle_file.close()

            print('Loading Videos ...')
            if self.mode == Constants.mode_open_clip_vivit:
                img_mod = Constants.mode_open_clip
                vid_mod = Constants.mode_vivit
            else:
                print('ERROR')
                exit(0)
            self.museums = json.load(open(self.data_museum_path, 'r'))
            if os.path.exists(f'data/videos_tensors_{img_mod}.pth'):
                self.videos_img = torch.load(f"data/videos_tensors_{img_mod}.pth")
            else:
                list_vids = list()
                for m in self.museums:
                    for r in m['rooms']:
                        list_vids.extend(m['rooms'][r])
                list_vids = list(set(list_vids))
                self.videos_img = {v: torch.load(self.data_frames_path_img + os.sep + v + '.pt') for v in list_vids}

                torch.save(self.videos_img, f"data/videos_tensors_{img_mod}.pth")

            if os.path.exists(f'data/videos_tensors_{vid_mod}.pth'):
                self.videos_vid = torch.load(f"data/videos_tensors_{vid_mod}.pth")
            else:
                list_vids = list()
                for m in self.museums:
                    for r in m['rooms']:
                        list_vids.extend(m['rooms'][r])
                list_vids = list(set(list_vids))
                self.videos_vid = {v: torch.load(self.data_frames_path_vid + os.sep + v + '.pt') for v in list_vids}

                torch.save(self.videos_vid, f"data/videos_tensors_{vid_mod}.pth")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.mem:
            desc_tensor = self.descs[index]
            list_vids_museum_img = list()
            list_vids_museum_vid = list()
            for r in self.museums[index]['rooms']:
                list_vids_museum_img.append([self.videos_img[self.museums[index]['rooms'][r][x]] for x in range(len(self.museums[index]['rooms'][r]))])
                list_vids_museum_vid.append([self.videos_vid[self.museums[index]['rooms'][r][x]] for x in range(len(self.museums[index]['rooms'][r]))])
        else:
            desc_tensor = torch.load(self.description_path + os.sep + self.samples[index] + '.pt')
            list_vids_museum_img = list()
            list_vids_museum_vid = list()
            for r in self.museums[index]['rooms']:
                list_vids_museum_img.append([torch.load(self.data_frames_path_img + os.sep + self.museums[index]['rooms'][r][x] + '.pt')
                                         for x in range(len(self.museums[index]['rooms'][r]))])
                list_vids_museum_vid.append([torch.load(self.data_frames_path_vid + os.sep + self.museums[index]['rooms'][r][x] + '.pt')
                                         for x in range(len(self.museums[index]['rooms'][r]))])

        return desc_tensor, list_vids_museum_img, list_vids_museum_vid
