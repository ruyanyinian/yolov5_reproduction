'''
@File    :   commom.py
@Time    :   2021/11/24 10:56:10
@Author  :   qinyu
@Contact :   qinyufight123@126.com
@License :   * Do Fuck You Want *
@Desc    :   network的公共接口层
'''


import torch
import torch.nn as nn


def autopad(k, p=None):
    """
    在stride=1的情况下, 为了保证输入在卷积后和卷积层的尺寸是一样的, 引入autopad
    功能类似于tf的tf.nn.conv2d中的padding='SAME'的功能。
    假设特征图是5×5, 但是在kernel=3×3的尺寸下, 保证卷积后的特征图也是5×5, 那么4条边应该扩充1条边
    整个特征图达到了7×7的尺寸。那么1就可以通过 kernel_size // 2 的操作得到
    Args:
        k (int, or tuple or list): kernel size
        p ([type], optional): [description]. Defaults to None.

    Returns:
        p (int): 扩充p条边
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    """标准的卷积操作层"""
    def __init__(self, c_in, c_out, k=1, s=1, g=1, p=None, act=True):
        """
        bn + conv + activation
        Args:
            c_in (int), channel_in 此参数用于conv的channel_in的指定
            c_out (int), channel_out 此参数用于conv的channel_out的指定
            k (int): 此参数用于conv的kernel_size指定
            s (int): 此参数用于conv的stride的指定
            g (int, optional): 此参数用于groups参数的指定. Defaults to 1.
            p (int or None, optional): 此参数用于conv的padding的指定. Defaults to None.
            act (nn.Module的对象 or bool, optional): 是否进行激活层. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, padding=autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        # 默认激活函数选择silu, 也可以自己选择不同的激活函数(必须是nn.Module的对象) 否则不做激活函数。这里用nn.Identity恒等映射替代
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    


if __name__ == '__main__':
    # 测试一些conv
    pass