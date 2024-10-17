import os
import pathlib
import numpy as np
import torch
import pytorch_lightning as pl
import open3d as o3d

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from pytorch3d.ops import knn_points

from .transforms import NormalizeUnitSphere, RandomScale, RandomRotate


class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, transform=None):
        super().__init__()

        self.pcl_dir = os.path.join(root, dataset, split)
        self.gt_path = str(os.path.join(root, dataset, 'gt')) if split == 'test' else None
        self.transform = transform
        self.pointcloud_names = []
        self.pointclouds = []
        self.masks = []
        self.labels = []
        # 先将所有的点云存入内存
        for fn in os.listdir(self.pcl_dir):
            # 对文件进行判断
            if fn[-3:] != 'pcd':
                continue

            pcl_path = os.path.join(self.pcl_dir, fn)

            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)

            # 读取点云坐标
            pcl = o3d.io.read_point_cloud(pcl_path)
            pcl = torch.FloatTensor(np.array(pcl.points, dtype=np.float32))

            # 存入当前点云的文件名
            self.pointcloud_names.append(fn[:-4])

            if split == 'train':
                # 训练样本全是正常数据
                # 直接存入点云坐标，逐点标签，整体标签
                self.pointclouds.append(pcl)
                self.masks.append(np.zeros(pcl.shape[0]))
                self.labels.append(0)
            elif split == 'test':
                if ('good' or 'positive') in fn[:-4]:
                    self.pointclouds.append(pcl)
                    self.masks.append(np.zeros(pcl.shape[0]))
                    self.labels.append(0)
                else:
                    filename = pathlib.Path(fn).stem
                    txt_path = os.path.join(self.gt_path, filename + '.txt')
                    pcd = np.genfromtxt(txt_path, delimiter=" ")

                    pcl = torch.FloatTensor(np.array(pcd[:, :3], dtype=np.float32))
                    self.pointclouds.append(pcl)
                    self.masks.append(pcd[:, 3])
                    self.labels.append(1)
        print(f'[INFO] Loaded dataset {dataset}')

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx],
            'mask': self.masks[idx],
            'label': self.labels[idx]
        }

        if self.transform is not None:
            data = self.transform(data)
        return data


def make_patches_for_pcl_pair(pcl_A, pcl_B, patch_size, num_patches, ratio):
    """
    Args:
        pcl_A:  The first point cloud, (N, 3).
        pcl_B:  The second point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_A.size(0)
    seed_idx = torch.randperm(N)[:num_patches]  # (P, )
    seed_pnts = pcl_A[seed_idx].unsqueeze(0)  # (1, P, 3)
    _, _, pat_A = knn_points(seed_pnts, pcl_A.unsqueeze(0), K=patch_size, return_nn=True)
    pat_A = pat_A[0]  # (P, M, 3)
    _, _, pat_B = knn_points(seed_pnts, pcl_B.unsqueeze(0), K=int(ratio * patch_size), return_nn=True)
    pat_B = pat_B[0]
    return pat_A, pat_B


def make_patches_for_pcl(data, patch_size, num_patches):
    """
    Args:
        data: {'pcl_clean': xx, 'label': xx, ...}
        pcl:  The first point cloud, (N, 3).
        patch_size:   Patch size M.
        num_patches:  Number of patches P.
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    pcl = data['pcl_clean']
    N = pcl.size(0)
    # randperm把点云的顺序随机打乱，取前P个作为seed
    seed_idx = torch.randperm(N)[:num_patches]  # (P, )
    seed_pnts = pcl[seed_idx].unsqueeze(0)  # (1, P, 3)
    # 以seed为中心，用knn找近邻，k = patch_size
    _, _idx, pat = knn_points(seed_pnts, pcl.unsqueeze(0), K=patch_size, return_nn=True)       # pat [1, P, M, 3]
    mask = data['mask'][_idx]
    # 判断这个patch是否异常，如果至少有一个点异常，整个patch异常
    label = 1 if np.any(mask == 1) else 0

    return pat[0], mask[0], label  # (P, M, 3), (P, M), (1)


class PatchDataset(Dataset):

    def __init__(self, datasets, patch_ratio, on_the_fly=True, patch_size=1000, num_patches=100, transform=None):
        super().__init__()
        self.datasets = datasets
        # self.len_datasets = sum([len(dset) for dset in datasets])
        self.len_datasets = len(datasets)       # 点云的个数（obj的个数）
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.on_the_fly = on_the_fly
        self.transform = transform
        self.patches = []
        # Initialize
        if not on_the_fly:
            self.make_patches()

    def make_patches(self):
        for dataset in tqdm(self.datasets, desc='MakePatch'):
            for data in tqdm(dataset):
                pat_noisy, pat_clean = make_patches_for_pcl_pair(
                    data['pcl_noisy'],
                    data['pcl_clean'],
                    patch_size=self.patch_size,
                    num_patches=self.num_patches,
                    ratio=self.patch_ratio
                )  # (P, M, 3), (P, rM, 3)
                for i in range(pat_noisy.size(0)):
                    self.patches.append((pat_noisy[i], pat_clean[i],))

    def __len__(self):
        if not self.on_the_fly:
            return len(self.patches)
        else:
            return self.len_datasets * self.num_patches

    def __getitem__(self, idx):
        if self.on_the_fly:
            pcl_dset = self.datasets
            pcl_data = pcl_dset[idx % len(pcl_dset)]
            # 每次只从点云(obj)中随机取一个patch，即从随机打乱的patch列表中取第一个
            pcl_clean, mask, label = make_patches_for_pcl(pcl_data, patch_size=self.patch_size, num_patches=self.num_patches)
            data = {
                'pcl_clean': pcl_clean[0],
                'mask': mask[0],
                'label': label
            }
            if self.transform is not None:
                data = self.transform(data)
                del data['noise_std']
        else:
            data = {
                'pcl_noisy': self.patches[idx][0].clone(),
                'pcl_clean': self.patches[idx][1].clone(),
            }

        return data


class ADDataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super(ADDataModule, self).__init__()
        self.cfg = cfg

    def train_dataloader(self):
        transforms = [
            NormalizeUnitSphere(),
            RandomScale([0.8, 1.2]),
        ]
        if self.cfg.aug_rotate:
            transforms += [
                RandomRotate(axis=0),
                RandomRotate(axis=1),
                RandomRotate(axis=2),
            ]
        transforms = Compose(transforms)

        pc_datasets = PointCloudDataset(root=self.cfg.dataset_root, dataset=self.cfg.dataset, split='train',
                                        transform=transforms)

        train_dset = PatchDataset(datasets=pc_datasets, patch_size=self.cfg.patch_size,
                                        num_patches=self.cfg.num_patches, patch_ratio=1.0, on_the_fly=True,
                                        transform=None)

        return DataLoader(train_dset, batch_size=self.cfg.train_batch_size, num_workers=self.cfg.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        transforms = Compose([NormalizeUnitSphere()])
        pc_datasets = PointCloudDataset(root=self.cfg.dataset_root, dataset=self.cfg.dataset, split='test',
                                     transform=transforms)
        val_dset = PatchDataset(datasets=pc_datasets, patch_size=self.cfg.patch_size,
                                        num_patches=self.cfg.num_patches, patch_ratio=1.0, on_the_fly=True,
                                        transform=None)


        return DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)


if __name__ == "__main__":
    root = '/home/jett/Data/Datasets/Real3D-AD-PCD'
    dataset = 'airplane'
    split = 'test'

    transforms = [
        NormalizeUnitSphere(),
        RandomScale([0.8, 1.2]),
    ]
    transforms += [
        RandomRotate(axis=0),
        RandomRotate(axis=1),
        RandomRotate(axis=2),
    ]
    transforms = Compose(transforms)

    pc_datasets = PointCloudDataset(root=root, dataset=dataset, split=split,
                                 transform=transforms)

    train_dset = PatchDataset(datasets=pc_datasets, patch_size=1024,
                              num_patches=100, patch_ratio=1.0, on_the_fly=True,
                              transform=None)

    train_loader = DataLoader(train_dset, batch_size=8, num_workers=4,
                          shuffle=True)
    num = shit = 0
    for data in train_loader:
        print(data['label'])
        num+=1