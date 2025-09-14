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


def prepare_input(output_data, list_nums):
    input_model = list()
    idx_nvr = 0
    for x in list_nums:
        input_model.append(output_data[idx_nvr:idx_nvr + x])
        idx_nvr += x
    input_model = pad_sequence(input_model, batch_first=True)
    input_model = torch.transpose(input_model, 1, 2)

    return input_model


def set_seed(seed_num):
    np.random.seed(seed_num)
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def start_train(args):
    set_seed(seed_num=42)

    mode = Constants.mode_mobile_clip

    approach_name = mode
    output_feature_size = 256

    is_bidirectional = True
    model_desc = GRUNet(hidden_size=output_feature_size, num_features=512, is_bidirectional=is_bidirectional)
    model_video = VideoModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=512)
    model_room = RoomModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=512)
    model_museum = MuseumModel(in_channels=512, out_channels=512, kernel_size=3, feature_size=256)

    cont_loss = train_utility.LossContrastive(name=approach_name, patience=25, delta=0.0001)

    num_epochs = 50
    batch_size = args.batch_size

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device = ', device)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    model_desc.to(device=device)
    model_video.to(device=device)
    model_room.to(device=device)
    model_museum.to(device=device)

    #     data section
    train_indices, val_indices, test_indices = train_utility.retrieve_indices()

    descriptions_path, videos_path, museum_path = train_utility.get_entire_data(mode=mode)

    dataset = DescriptionScene(data_description_path=descriptions_path, data_video_path=videos_path,
                               data_museum_path=museum_path, mode=mode, mem=True)
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    test_subset = Subset(dataset, test_indices.tolist())

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, num_workers=4, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=4, worker_init_fn=seed_worker, generator=g)

    '''Train Procedure'''
    params = list(model_desc.parameters()) + list(model_video.parameters()) + list(model_room.parameters()) + list(model_museum.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    step_size = 27
    gamma = 0.75
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    best_r10 = 0
    print('Train procedure ...')
    for _ in tqdm(range(num_epochs)):

        if not cont_loss.is_val_improving():
            print('Early Stopping !!!')
            break

        total_loss_train = 0
        total_loss_val = 0
        num_batches_train = 0
        num_batches_val = 0

        output_description_val = torch.empty(len(val_indices), output_feature_size)
        output_museum_val = torch.empty(len(val_indices), output_feature_size)

        max_loss = 0
        # descs, frames, num_videos_in_room, num_rooms
        for i, (data_desc, data_frame, data_nvr, data_r) in enumerate(tqdm(train_loader, total=(len(train_indices) // batch_size))):
            data_desc = data_desc.to(device)
            data_frame = data_frame.to(device)

            optimizer.zero_grad()

            output_desc = model_desc(data_desc)
            output_video = model_video(data_frame)

            input_room_model = prepare_input(output_data=output_video, list_nums=data_nvr)
            # print(input_room_model.shape)
            output_room = model_room(input_room_model, data_nvr)

            input_museum_model = prepare_input(output_data=output_room, list_nums=data_r)
            output_museum = model_museum(input_museum_model, data_r)

            multiplication_dp = train_utility.cosine_sim(output_desc, output_museum)

            loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

            if loss_contrastive.item() > max_loss:
                max_loss = loss_contrastive.item()

            loss_contrastive.backward()

            optimizer.step()

            total_loss_train += loss_contrastive.item()
            num_batches_train += 1

        scheduler.step()
        print(scheduler.get_last_lr())
        print('total_loss_train', total_loss_train)
        epoch_loss_train = total_loss_train / num_batches_train

        # Validation Procedure
        model_desc.eval()
        model_museum.eval()
        with torch.no_grad():
            for j, (data_desc, data_frame, data_nvr, data_r) in enumerate(val_loader):

                data_desc = data_desc.to(device)
                data_frame = data_frame.to(device)

                output_desc = model_desc(data_desc)
                output_video = model_video(data_frame)

                input_room_model = prepare_input(output_data=output_video, list_nums=data_nvr)
                # print(input_room_model.shape)
                output_room = model_room(input_room_model, data_nvr)

                input_museum_model = prepare_input(output_data=output_room, list_nums=data_r)
                output_museum = model_museum(input_museum_model, data_r)

                initial_index = j * batch_size
                final_index = (j + 1) * batch_size
                if final_index > len(val_indices):
                    final_index = len(val_indices)

                output_description_val[initial_index:final_index, :] = output_desc
                output_museum_val[initial_index:final_index, :] = output_museum

                multiplication_dp = train_utility.cosine_sim(output_desc, output_museum)

                loss_contrastive = cont_loss.calculate_loss(multiplication_dp)

                total_loss_val += loss_contrastive.item()
                num_batches_val += 1

            epoch_loss_val = total_loss_val / num_batches_val

            print('Loss Train', epoch_loss_train)
            cont_loss.on_epoch_end(epoch_loss_train, train=True)
            print('Loss Val', epoch_loss_val)
            cont_loss.on_epoch_end(epoch_loss_val, train=False)

        r1, r5, r10, _, _, _, _, _, _, _ = train_utility.evaluate(output_description=output_description_val,
                                                                  output_scene=output_museum_val, section='val')

        model_desc.train()
        model_museum.train()

        if r10 > best_r10:
            best_r10 = r10
            train_utility.save_best_model(approach_name, model_video.state_dict(), model_room.state_dict(), model_museum.state_dict(), model_desc.state_dict())

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
            # print(input_room_model.shape)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="num of points",
                        default=64,
                        required=False)
    parser.add_argument("--lr", type=float, help="Learning Rate",
                        default=0.0007,
                        required=False)
    args = parser.parse_args()
    start_train(args)
