import os
import sys

import numpy as np

from models.padflow import PadFlow

sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl

from torch import Tensor
from argparse import ArgumentParser

from dataset.dataset import ADDataModule
from utils.callback import TimeTrainingCallback
from utils.lightning import LightningProgressBar
from utils.log import print_progress_log


class TrainerModule(pl.LightningModule):

    def __init__(self, cfg):
        super(TrainerModule, self).__init__()

        self.network = PadFlow(pc_channel=3)

        self.epoch = 0
        self.cfg = cfg

        self.threshold = 0.5

    def forward(self, p: Tensor, **kwargs):
        return self.network(p, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.cfg.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        pcl_clean = batch['pcl_clean']
        x, logpx = self(pcl_clean)

        loss = logpx

        self.log('logpx', logpx, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        return loss

    def validation_step(self, batch, batch_idx):
        pcl_clean, gt_label = batch['pcl_clean'], batch['label']
        x, logpx = self(pcl_clean)
        # pcl_denoised = patch_denoise(self, pcl_noisy.squeeze(), patch_size=1024)  # Fix patch size
        #
        # return {
        #     'denoised': pcl_denoised,
        #     'clean': pcl_clean.squeeze(),
        # }
        label = 0 if logpx < self.threshold else 1
        return {
            'logpx': logpx,
            'label': label,
            'gt_label': gt_label,
        }

    def on_validation_end(self, batch):
        label = torch.stack([x['label'] for x in batch])
        gt_label = torch.stack([x['gt_label'] for x in batch])

        accuracy = np.mean(label == gt_label)

        extra = []

        print_progress_log(self.epoch, {'acc': accuracy}, extra=extra)
        self.epoch += 1


# 模型训练参数
def model_specific_args():
    parser = ArgumentParser()

    # Network
    parser.add_argument('--net', type=str, default='pad-flow')
    # Optimizer and scheduler
    parser.add_argument('--learning_rate', default=2e-3, type=float)
    parser.add_argument('--sched_patience', default=10, type=int)
    parser.add_argument('--sched_factor', default=0.5, type=float)
    parser.add_argument('--min_lr', default=1e-6, type=float)
    # Training
    parser.add_argument('--max_epoch', default=50, type=int)
    parser.add_argument('--seed', default=2024, type=int)

    return parser

# 数据集训练参数
def dataset_specific_args():
    parser = ArgumentParser()

    parser.add_argument('--dataset_root', default='/home/jett/Data/Datasets/Real3D-AD-PCD', type=str)
    parser.add_argument('--dataset', default='airplane', type=str)
    parser.add_argument('--aug_rotate', default=True, choices=[True, False])
    parser.add_argument('--patch_size', type=int, default=1024)
    parser.add_argument('--num_patches', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', default=4, type=int)

    return parser


def train(phase='Train', checkpoint_path=None, begin_checkpoint=None):
    comment = 'Real3D-AD'
    cfg = model_specific_args().parse_args()
    pl.seed_everything(cfg.seed)

    dataset_cfg = dataset_specific_args().parse_args()
    datamodule = ADDataModule(dataset_cfg)

    trainer_config = {
        'default_root_dir': './runs/',
        'accelerator': 'auto',
        'fast_dev_run': False,
        'max_epochs': 150,  # cfg.max_epoch,
        'precision': 32,  # 32, 16, 'bf16'
        'gradient_clip_val': 1e-3,
        'deterministic': False,
        'num_sanity_val_steps': 0,  # -1,  # -1 or 0
        'enable_checkpointing': False,
        'callbacks': [TimeTrainingCallback(), LightningProgressBar()],
        # 'profiler'             : "pytorch",
    }

    module = TrainerModule(cfg)
    trainer = pl.Trainer(**trainer_config)
    trainer.is_interrupted = False

    if phase == 'Train':
        if comment is not None:
            print(f'\nComment: \033[1m{comment}\033[0m')
        if begin_checkpoint is not None:
            state_dict = torch.load(begin_checkpoint)
            module.network.load_state_dict(state_dict)
            module.network.init_as_trained_state()

        trainer.fit(model=module, datamodule=datamodule)

        if checkpoint_path is not None and trainer_config['fast_dev_run'] is False and trainer.is_interrupted is False:
            if trainer_config["max_epochs"] > 10:
                save_path = checkpoint_path + f'-epoch{trainer_config["max_epochs"]}.ckpt'
                torch.save(module.network.state_dict(), save_path)
                print(f'Model has been save to \033[1m{save_path}\033[0m')
    else:  # Test
        state_dict = torch.load(begin_checkpoint)
        module.network.load_state_dict(state_dict)
        module.network.init_as_trained_state()
        trainer.test(model=module, datamodule=datamodule)



if __name__ == "__main__":
    checkpoint_path = 'runs/ckpt/PadFlow-Real3D-AD'

    train('Train', checkpoint_path, None)
