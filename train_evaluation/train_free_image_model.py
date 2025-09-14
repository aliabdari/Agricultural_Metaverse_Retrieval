import os
import numpy as np
import torch
import json
import pickle


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def aggregation_func(mod, x):
    if mod == 'Max':
        max_values_keepdim, _ = torch.max(x, dim=0, keepdim=True)
        return max_values_keepdim
    elif mod == 'Mean':
        return torch.mean(x, dim=0)
    elif mod == 'Median':
        median_values, indices = torch.median(x.to(torch.float32), dim=0)
        return median_values.to(torch.float16)


def calculate_accuracy(feature_set_1, feature_set_2):
    ranks = []
    for idx in range(feature_set_2.shape[0]):
        distances = np.array([euclidean_distance(feature_set_2[idx], x) for x in feature_set_1])
        sorted_indexes = np.argsort(distances)
        # sorted_array = feature_set_1[sorted_indexes]
        ranks.append(np.where(sorted_indexes == idx)[0].tolist()[0])
    ranks = np.array(ranks)
    n_q = feature_set_2.shape[0]
    r1 = 100 * len(np.where(ranks < 1)[0]) / n_q
    r5 = 100 * len(np.where(ranks < 5)[0]) / n_q
    r10 = 100 * len(np.where(ranks < 10)[0]) / n_q
    mrr = 100 * (sum(1 / (x + 1) for x in ranks) / len(ranks))

    ranks = np.array(ranks)
    print('r1', r1)
    print('r5', r5)
    print('r10', r10)
    print('Rank Median:', np.median(ranks) + 1)
    print('Rank Mean:', ranks.mean() + 1)
    print('MRR', mrr)


def start_process():
    museums = json.load(open('../museums.json', 'r'))

    indices = pickle.load(open('indices/indices.pkl', 'rb'))
    test_indices = indices['test'].tolist()

    fvrs = [('Mean', 'Mean', 'Mean'), ('Median', 'Mean', 'Mean'),
            ('Mean', 'Median', 'Mean'), ('Median', 'Median', 'Mean')]

    path_tensors_frames = '../features/open_clip_features/frames'
    path_tensors_descriptions = '../features/open_clip_features/descriptions/sentences'

    total_museums_frames_features = list()
    total_museums_descriptions_features = list()
    for fvr in fvrs:
        for idx_m in test_indices:
            m = museums[idx_m]
            # if idx_m + 1 != m['id']:
            #     print(idx_m, m)
            features_frames_museum = torch.zeros(len(m['rooms']), 512)
            # features_descriptions_museum = list()
            for idx_r, r in enumerate(m['rooms']):
                features_frames_rooms = torch.zeros(len(m['rooms'][r]), 512)
                # features_descriptions_rooms = list()
                for idx_v, v in enumerate(m['rooms'][r]):
                    features_frames = torch.load(path_tensors_frames + os.sep + v + '.pt', weights_only=True)
                    # features_descriptions = torch.load(path_tensors_descriptions + os.sep + v + '.pt', weights_only=True)
                    features_frames_rooms[idx_v, :] = aggregation_func(fvr[0], features_frames)
                    # features_descriptions_rooms.append(aggregation_func(fvr[0], features_descriptions))
                features_frames_museum[idx_r, :] = aggregation_func(fvr[1], features_frames_rooms)
                # features_descriptions_museum.extend(features_descriptions_rooms)
            total_museums_frames_features.append(aggregation_func(fvr[2], features_frames_museum))
            total_museums_descriptions_features.append(torch.mean(torch.load(path_tensors_descriptions + os.sep + 'Museum_' + str(idx_m) + '.pt', weights_only=True),0))
            # torch.save(aggregation_func(fvr[2], features_museum), path_output + os.sep + 'museum_' + str(idx_m) + '.pt')

        print(f'Retrieving Museums based on the Sentences: for {fvr}')
        calculate_accuracy(torch.stack(total_museums_frames_features), torch.stack(total_museums_descriptions_features))


if __name__ == '__main__':
    start_process()
