import torch
import os 
import argparse
from pathlib import Path
from utils.general import check_file, print_args

FILE = Path(__file__).resolve()
ROOT = FILE.parent


def train(hyp, opt):
    
    pass

def parse_opt(known=False):
    """
    命令行参数的设置
    Args:
        known (bool, optional): [是否解析已经存在的超参数]. Defaults to False.

    Returns:
        [opt]: [dict]
    """
    parser =  argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperpareters path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='dataset yaml file')
    parser.add_argument('--cfg', type=str, default='', help='model config yaml')
    # TODO(qinyu): 带填充相关的命令行参数
    opt = parser.parse_known_args()[0] if known else parser.parse_args() # 如果known=True的话, 即使在命令行输入错了没有的参数, 也一样可以进行解析
    return opt

def main(opt):
    """
    主函数: 主要功能如下
    1.检查一些文件是否符合要求
    2.重新训练/或者接着上一次断的地方开始训练
    3.多GPU/单GPU训练
    4.绘制结果
    Args:
        opt ([argparse.Namespace的对象]): [description]
    """

    # TODO(qinyu): 检查一些路径是否正确, 包括一些关于后缀名是否正确
    
    # 打印命令行参数
    print_args(FILE.stem, opt)
    train()
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)