import os
import random
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class HSIdataset(Dataset):
    def __init__(self, hsi_h5_dir, dist_h5_dir, data_dict, mode='train', aug=False, rand_crop=False, rand_map=False,
                 crop_size=32):
        self.data_dict = data_dict
        with h5py.File(hsi_h5_dir, 'r') as f:
            data = f['data'][:]
        self.data = data / data.max()
        if mode == 'train':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['train_label_map'][0]
        elif mode == 'val':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['val_label_map'][0]
        elif mode == 'test':
            with h5py.File(dist_h5_dir, 'r') as f:
                label_map = f['test_label_map'][0]
        self.label_map = label_map
        self.aug = aug
        self.rand_crop = rand_crop
        self.height, self.width = self.label_map.shape  # Get dimensions of label map
        self.crop_size = crop_size
        self.rand_map = rand_map

    def __getitem__(self, idx):
        if self.rand_map:
            label_map_t = self.get_rand_map(self.label_map)
        else:
            label_map_t = self.label_map
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0
        if self.rand_crop:
            flag = 0
            while flag == 0:
                x1 = random.randint(0, self.width - self.crop_size - 1)
                x2 = x1 + self.crop_size
                y1 = random.randint(0, self.height - self.crop_size - 1)
                y2 = y1 + self.crop_size

                if label_map_t[y1:y2, x1:x2].max() > 0:
                    flag = 1
            input_data = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]
        else:
            patch_info = self.data_dict[idx]
            x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
            input_data = self.data[y1:y2, x1:x2]
            target = label_map_t[y1:y2, x1:x2]
        if self.aug:
            input_data, target = self.random_flip_lr(input_data, target)
            input_data, target = self.random_flip_tb(input_data, target)
            input_data, target = self.random_rot(input_data, target)
        return (torch.from_numpy(input_data).float().permute(2, 0, 1).unsqueeze(dim=0),
                torch.from_numpy(target - 1).long())

    def __len__(self):
        return len(self.data_dict)

    @staticmethod
    def random_flip_lr(input_data, target):
        if np.random.randint(0, 2):
            h, w, d = input_data.shape
            index = np.arange(w, 0, -1) - 1
            return input_data[:, index, :], target[:, index]
        else:
            return input_data, target

    @staticmethod
    def random_flip_tb(input_data, target):
        if np.random.randint(0, 2):
            h, w, d = input_data.shape
            index = np.arange(h, 0, -1) - 1
            return input_data[index, :, :], target[index, :]
        else:
            return input_data, target

    @staticmethod
    def random_rot(input_data, target):
        rot_k = np.random.randint(0, 4)
        return np.rot90(input_data, rot_k, (0, 1)).copy(), np.rot90(target, rot_k, (0, 1)).copy()

    @staticmethod
    def get_rand_map(label_map, keep_ratio=0.6):
        label_map_t = label_map
        label_indices = np.where(label_map > 0)
        label_num = len(label_indices[0])
        shuffle_indices = np.random.permutation(int(label_num))
        dis_num = int(label_num * (1 - keep_ratio))
        dis_indices = (label_indices[0][shuffle_indices[:dis_num]], label_indices[1][shuffle_indices[:dis_num]])
        label_map_t[dis_indices] = 0
        return label_map_t


class HSIdatasettest(Dataset):
    def __init__(self, hsi_data, data_dict):
        self.HSI_data = hsi_data / hsi_data.max()
        self.data_dict = data_dict

    def __getitem__(self, idx):
        patch_info = self.data_dict[idx]
        x1, x2, y1, y2 = patch_info['x1'], patch_info['x2'], patch_info['y1'], patch_info['y2']
        input_data = self.HSI_data[y1:y2, x1:x2]
        return torch.from_numpy(input_data).float().permute(2, 0, 1).unsqueeze(dim=0), [x1, x2, y1, y2]

    def __len__(self):
        return len(self.data_dict)


def h5_dist_loader(data_dir):
    with h5py.File(data_dir, 'r') as f:
        height, width = f['height'][0], f['width'][0]
        category_num = f['category_num'][0]
        train_map, val_map, test_map = f['train_label_map'][0], f['val_label_map'][0], f['test_label_map'][0]
    return height, width, category_num, train_map, val_map, test_map


def get_patches_list(height, width, crop_size, label_map, patches_num, shuffle=True):
    patch_list = []
    count = 0
    if shuffle:
        while count < patches_num:
            x1 = random.randint(0, width - crop_size - 1)
            y1 = random.randint(0, height - crop_size - 1)
            x2, y2 = x1 + crop_size, y1 + crop_size
            if label_map[y1:y2, x1:x2].max() > 0:
                patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                patch_list.append(patch)
                count += 1
    else:
        slide_step = crop_size
        x1_list = list(range(0, width - crop_size, slide_step))
        y1_list = list(range(0, height - crop_size, slide_step))
        x1_list.append(width - crop_size)
        y1_list.append(height - crop_size)
        x2_list = [x + crop_size for x in x1_list]
        y2_list = [y + crop_size for y in y1_list]
        for x1, x2 in zip(x1_list, x2_list):
            for y1, y2 in zip(y1_list, y2_list):
                if label_map[y1:y2, x1:x2].max() > 0:
                    patch = {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2}
                    patch_list.append(patch)
    return patch_list


def build_dataset(cfg):
    data_root = cfg.DATASET.DATA_ROOT
    data_set = cfg.DATASET.DATA_SET
    crop_size = cfg.DATASET.CROP_SIZE
    data_list_dir = cfg.DATALOADER.DATA_LIST_DIR
    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.DATALOADER.BATCH_SIZE_TRAIN
    search_on = cfg.SEARCH.SEARCH_ON
    dist_dir = os.path.join(data_list_dir, '{}_dist_{}_train-{}_val-{}.h5'.
                            format(data_set,
                                   cfg.DATASET.DIST_MODE,
                                   float(cfg.DATASET.TRAIN_NUM),
                                   float(cfg.DATASET.VAL_NUM)))
    height, width, category_num, train_map, val_map, test_map = h5_dist_loader(dist_dir)

    if search_on:
        w_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        a_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM // 2, shuffle=True)
        v_data_list = get_patches_list(height, width, crop_size, val_map, cfg.DATASET.PATCHES_NUM, shuffle=False)
        dataset_w = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=w_data_list, mode='train')
        dataset_a = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=a_data_list, mode='train')
        dataset_v = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                               dist_h5_dir=dist_dir,
                               data_dict=v_data_list, mode='val')
        data_loader_w = torch.utils.data.DataLoader(
            dataset_w,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        data_loader_a = torch.utils.data.DataLoader(
            dataset_a,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        data_loader_v = torch.utils.data.DataLoader(
            dataset_v,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        return [data_loader_w, data_loader_a], data_loader_v
    else:
        tr_data_list = get_patches_list(height, width, crop_size, train_map, cfg.DATASET.PATCHES_NUM, shuffle=True)
        te_data_list = get_patches_list(height, width, crop_size, test_map, cfg.DATASET.PATCHES_NUM, shuffle=False)
        dataset_tr = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=tr_data_list, mode='train', aug=True, rand_crop=True, crop_size=crop_size)

        dataset_te = HSIdataset(hsi_h5_dir=os.path.join(data_root, '{}.h5'.format(data_set)),
                                dist_h5_dir=dist_dir,
                                data_dict=te_data_list, mode='test')
        data_loader_tr = torch.utils.data.DataLoader(
            dataset_tr,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        data_loader_te = torch.utils.data.DataLoader(
            dataset_te,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True)
        return data_loader_tr, data_loader_te
