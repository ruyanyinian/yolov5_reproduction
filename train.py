import torch
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent

def train():
    pass

def parse_opt(known=False):
    parser =  argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperpareters path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset yaml file')
    parser.add_argument('--cfg', type=str, default='', help='model config yaml')
    # TODO(qinyu): 带填充相关的命令行参数
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt
def main(opt):
    pass

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)