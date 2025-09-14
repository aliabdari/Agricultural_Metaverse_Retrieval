import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Type
import logging
# from layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
#     create_grouper, furthest_point_sample, random_sample, three_interpolation, get_aggregation_feautres


class Combiner(nn.Module):
    def __init__(self, input_features=256, output_features=256, num_inputs=4, hidden_dim=256):
        super(Combiner, self).__init__()
        self.total_input_features = input_features * num_inputs
        self.fc1 = nn.Linear(self.total_input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_features)
        self.activation = nn.ReLU()

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=-1)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x


class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(OneDimensionalCNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(out_channels, feature_size)

    def forward(self, x, list_length=None):
        x = x.to(torch.float32)
        x = self.conv(x)
        # remove the effect of the padding
        # if list_length is not None:
        #     for item_idx in range(x.shape[0]):
        #         x1[item_idx, :, list_length[item_idx]:] = 0
        list_avg_tensors = list()
        for item_idx in range(x.shape[0]):
            list_avg_tensors.append(torch.mean(x[item_idx, :, :list_length[item_idx]], dim=1))
        x = torch.stack(list_avg_tensors, dim=0)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class GRUNet(nn.Module):
    def __init__(self, hidden_size, num_features, is_bidirectional=False):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_size, batch_first=True,
                          bidirectional=is_bidirectional)
        self.is_bidirectional = is_bidirectional

    def forward(self, x):
        x = x.to(torch.float32)
        _, h_n = self.gru(x)
        if self.is_bidirectional:
            return h_n.mean(0)
        return h_n.squeeze(0)


# def get_reduction_fn(reduction):
#     reduction = 'mean' if reduction.lower() == 'avg' else reduction
#     assert reduction in ['sum', 'max', 'mean']
#     if reduction == 'max':
#         pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
#     elif reduction == 'mean':
#         pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
#     elif reduction == 'sum':
#         pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
#     return pool


class VideoModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(VideoModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512*32, feature_size)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.conv(x)
        # remove the effect of the padding
        # if list_length is not None:
        #     for item_idx in range(x.shape[0]):
        #         x[item_idx, :, list_length[item_idx]:] = 0
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RoomModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(RoomModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.adaptive_model = nn.AdaptiveAvgPool1d(1)
        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, feature_size)

    def forward(self, x, list_length):
        x = x.to(torch.float32)
        x = self.conv(x)
        # remove the effect of the padding
        # if list_length is not None:
        #     for item_idx in range(x.shape[0]):
        #         x[item_idx, :, list_length[item_idx]:] = 0
        list_avg_tensors = list()
        for item_idx in range(x.shape[0]):
            list_avg_tensors.append(torch.mean(x[item_idx, :, :list_length[item_idx]], dim=1))
        x = torch.stack(list_avg_tensors, dim=0)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MuseumModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_size):
        super(MuseumModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)

        self.bn = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512, feature_size)

    def forward(self, x, list_length):
        x = x.to(torch.float32)
        x = self.conv(x)
        list_avg_tensors = list()
        for item_idx in range(x.shape[0]):
            list_avg_tensors.append(torch.mean(x[item_idx, :, :list_length[item_idx]], dim=1))
        x = torch.stack(list_avg_tensors, dim=0)
        x = self.bn(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class CombineNetLateFusion(nn.Module):
    def __init__(self):
        super(CombineNetLateFusion, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CombineNetEarlyFusion(nn.Module):
    def __init__(self):
        super(CombineNetEarlyFusion, self).__init__()
        self.fc1 = nn.Linear(1280, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 512)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
