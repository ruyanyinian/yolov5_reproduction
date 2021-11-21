"""
General utils
"""
import logging
import glob
import re
from os import sep 
from pathlib import Path

def set_logging(name, verbose=True):
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)


def print_args(name, opt):
    LOGGER.info(colorstr(f'{name}: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))




def check_file():
    pass


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def increment_path(path):
    """
    对传入的path逐渐递增, 比如E:\\Data\\image_detection\\coco\\output\\runs\\exp --> E:\\Data\\image_detection\\coco\\output\\runs\\exp1
    E:\\Data\\image_detection\\coco\\output\\runs\\exp1 --> ...\\exp2
    或者对..haha.txt --> haha1.txt
    Args:
        path (str): [字符串路径]

    Returns:
        [path]: [Path的对象]
    """
    path = Path(path)
    if path.exists():
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '') # 把路径的后缀和文件分开
        dirs = glob.glob(f'{path}*') # list
        # E:\\Data\\image_detection\\coco\\output\\runs\\exp匹配的是None,因为里面没有数字exp1匹配的是1
        searches = [re.search(rf'%s(\d+)' % path.stem, d) for d in dirs if d] 
        i = [int(s.group()[0]) for s in searches if s]
        increment_i = max(i) + 1 if i else 1
        path = Path(f'{path}{increment_i}{suffix}')
    return path

if __name__ == '__main__':
    string = colorstr('blue', 'hello world')
    print(string)
    # path = 'E:\\Data\\image_detection\\coco\\output\\runs\\exp'
    # increment_path(path)