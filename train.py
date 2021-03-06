'''
@File    :   train.py
@Time    :   2021/11/24 10:55:23
@Author  :   qinyu
@Contact :   qinyufight123@126.com
@License :   * Do Fuck You Want *
@Desc    :   训练流程相关
'''


import torch
import os 
import argparse
import yaml
from pathlib import Path
from utils.general import LOGGER, check_file, check_yaml, colorstr, increment_path, print_args
from models.yolo import Model


FILE = Path(__file__).resolve()
ROOT = FILE.parent


def train(hyp, opt):
    save_dir = Path(opt.save_dir)
    
    # 目录相关
    w = save_dir / 'weights'
    w.parent.mkdir(parents=True, exist_ok=True) # exist_ok是如果存在目录,则不抛出异常
    last, best = w / 'last.pt', w / 'best.pt'

    # 读取相关的hyps.yaml超参数
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f) # dict 
    LOGGER.info(colorstr('Hyperparemeter: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # 保存opt和hyp
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)    
    
    # model
    model = Model()
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
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model config yaml')
    parser.add_argument('--project', type=str, default=ROOT / 'output/train', help='save everything in project/name')
    parser.add_argument('--name', type=str, default=ROOT / 'exp', help='save everything in project/name')
    # TODO(qinyu): 带填充相关的命令行参数
    opt = parser.parse_known_args()[0] if known else parser.parse_args() # 如果known=True的话, 即使在命令行输入错了没有的参数, 也一样可以进行解析
    return opt

def main(opt):
    """
    主函数: 主要功能如下
    1.检查一些文件是否符合要求, 并且对opt进行更新
    2.重新训练/或者接着上一次断的地方开始训练
    3.多GPU/单GPU训练
    4.绘制结果
    Args:
        opt ([argparse.Namespace的对象]): [description]
    """
    # 检查路径是否正确, 并且更新路径
    opt.data, opt.hyp, opt.cfg = check_file(opt.data), check_yaml(opt.hyp), check_yaml(opt.cfg)
    
    # 打印命令行参数
    print_args(FILE.stem, opt)
    opt.save_dir = str(increment_path(Path()))
    train(opt.hyp, opt)
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    