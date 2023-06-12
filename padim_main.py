import argparse
from src.modules.padim_high import PaDiM as hPaDiM
from src.modules.padim_efficient import PaDiM as ePaDiM


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_category', choices = ['Rivet_scr_all','Rivet_scr','Rivet_scr_resize'],type=str, default='Rivet_scr')
    parser.add_argument('--data_path', type=str, default='/home/oshita/padim/datasets')
    parser.add_argument('--save_path', type=str, default='./output_padim')
    parser.add_argument('--use_layers', type=str,default='1-2')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'resnet50'], default='wide_resnet50_2')
    # parser.add_argument('--arch', type=str, choices=['EfficientNet-B4','EfficientNet-B5','EfficientNet-B7'], default='EfficientNet-B7')
    parser.add_argument('--Rd', type=int, default=550)
    parser.add_argument('--non_Rd', action='store_true')
    parser.add_argument('--seed', choices=[64,128,256,512,1024],type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()

    padim = hPaDiM(config)
    # padim = ePaDiM(config)
    padim.test()

