# -*- coding: utf-8 -*-
# @Time    : 6/14/18 10:36 AM
# @Author  : Liu jiandong
# @FileName: decompose.py
# @Blog    ：http://meepoljd.github.io

import numpy as np
# from nmf import non_negative_factorization
from sklearn.decomposition import non_negative_factorization

"""
mu算法不会出现真正的0值，所以这里改为cd迭代算法
TODO: alpha与l1_ratio这里取0.5的默认值，后续测试一个更好的值。
"""

def decompose(spec, k, max_iter=6000, alpha=0.5):
    """
    basic NMF tool, use it to get W and H

    Example:
    >>> V = 10 * np.random.rand(100, 3000)
    >>> W, H = decompose(V, 50)

    :param spec:
    :param k:
    :param max_iter:
    """

    _dic, _act, n_iter = non_negative_factorization(spec,
                                               n_components=k,
                                               solver='cd',
                                               alpha=alpha,
                                               l1_ratio=1,
                                               max_iter=max_iter)
    return _dic, _act


def decompose_with_dict(spec, dic, max_iter=6000, alpha=0.5):
    """
    get H with V and W

    Example:
    >>> V = 10*np.random.rand(100, 200)
    >>> W, H = decompose(V, k=50)
    >>> H2 = decompose_with_dict(V, W)

    :param spec:
    :param dic:
    :param max_iter
    :param alpha
    :param l1_rate
    :return:
    """
    k = dic.shape[1]
    _act, _, n_iter = non_negative_factorization(np.transpose(spec),
                                            H=np.transpose(dic),
                                            update_H=False,
                                            alpha=alpha,
                                            l1_ratio=1,
                                            n_components=k,
                                            solver='cd',
                                            max_iter=max_iter)
    return np.transpose(_act)


if __name__ == '__main__':
    V = 10 * np.random.rand(100, 3000)
    W, H = decompose(V, 50)
    print(W)
