import argparse
# from src.modules.padim import PaDiM
# from src.modules.padim_cupy import PaDiM as cPaDiM
from src.modules.padim1_high import PaDiM as hPaDiM


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_category', type=str, default='bottle')
    parser.add_argument('--data_path', type=str, default='/home/data/mvtec')
    parser.add_argument('--save_path', type=str, default='./output_padim')
    parser.add_argument('--use_layers', type=str, default='1-2-3')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'resnet50'], default='wide_resnet50_2')
    parser.add_argument('--Rd', type=int, default=550)
    parser.add_argument('--non_Rd', action='store_true')
    parser.add_argument('--seed', type=int, default=1024)
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()

    # padim = PaDiM(config)
    # padim = cPaDiM(config)
    padim = hPaDiM(config)
    padim.test()
