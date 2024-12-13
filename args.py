import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="FreeNet")
    parser.add_argument(
        '--model_name',
        default='CSCN',
        type=str,
        help='name of model')
    parser.add_argument(
        '--seed',
        default=1,
        type=int,
        help='random seed')
    parser.add_argument(
        '--config',
        default=dict(
            in_channels=32,
            num_classes=24,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ),
        help='config of FreeNet')
    parser.add_argument(
        '--epoch_num',
        default=50,
        type=int,
        help='number of epoch')
    parser.add_argument(
        '--BatchSize',
        default=2,
        type=int,
        help='BatchSize')
    parser.add_argument(
        '--num_workers',
        default=2,
        type=int,
        help='num_workers')
    parser.add_argument(
        '--data_root',
        default='/WHU/',
        type=str,
        help='data dir')
    parser.add_argument(
        '--lr',
        default=1e-4,
        type=float,
        help='LearningRate')
    parser.add_argument(
        '--save_dir',
        default='/experiment/',
        type=str,
        help='experiment save dir')
    parser.add_argument(
        '--save_image_dir',
        default='/experiment/images/',
        type=str,
        help='images save dir')
    parser.add_argument(
        '--local_rank',
        default=os.getenv('LOCAL_RANK', -1),
        type=int,
        help="number of cpu threads to use during batch generation")
    parser.add_argument(
        '--update_lr',
        default=False,
        type=bool,
        help='if update learning rate')
    parser.add_argument(
        '--num',
        default=1,
        type=int,
        help='num')
    parser.add_argument(
        '--lamuda',
        default=0.5,
        type=float,
        help='ratio of loss')

    args = parser.parse_args()

    return args