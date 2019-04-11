# -*- coding: utf-8 -*
import numpy as np
from utilsm import get_spec
from decompose import decompose, decompose_with_dict

def is_in_dict(x, dic, tol=0.01, by='max_err'):
    """
    用来判断一个向量是否包含于一个字典所描述的子空间中

    Example:
    >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]])
    >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.double)
    >>> v = W1[:, 0]
    >>> is_in_dict(v, W2)
    True

    :param x: 待测试向量
    :param dic: 描述子空间字典
    :param tol: 可接受的阈值
    :param by: 判断方式，max_err,mean_err,act
    """
    if x is not np.matrixlib.defmatrix.matrix:
        vec = np.mat(x).T
    else:
        vec = x
    if vec.shape[1]!=1:
        vec = vec.T
    h = decompose_with_dict(vec, dic)
    re_ = np.dot(dic, h)
    err = np.square(vec - re_).reshape(-1, 1)
    if by=='max_err':
        return np.max(err) < tol
    elif by=='mean_err':
        return np.mean(err) < tol
    elif by=='act':
        return np.any(h<=tol)
    else:
        raise Exception('get wrong param by')

def depart_dic(index, dic):
    """
    将一个字典指定列的向量抽离，返回抽离向量与剩余部分矩阵

    Example:
    >>> W = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.double)
    >>> depart_dic(1, W)
    (matrix([[0.],
            [1.],
            [0.],
            [0.]]), matrix([[1., 0.],
            [0., 0.],
            [0., 1.],
            [0., 0.]]))

    :param index: index of target base
    :param dic:dic to depart
    """
    vec = np.mat(dic[:, index]).T
    rest_dic = np.column_stack([dic[:, i] for i in filter(lambda x:x!= index, range(dic.shape[1]))])
    return vec.T, rest_dic

def check_dic(dic, tol=0.01, by='max_err'):
    """
    用来检查一个字典是否有多余的内容

    Example:
    >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]])
    >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.double)
    >>> W = np.column_stack([W1, W2])
    >>> check_dic(W)
    matrix([[0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.]])
    """
    index = 0
    w = dic
    while index < w.shape[1]:
        v, rest = depart_dic(index, w)
        if is_in_dict(v, rest):
            w = rest
        else:
            index += 1
    return w

def get_child(x, dic, tol=0.001):
    """
    获取一个向量所包含的字典中的成分

    Example:
    >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]])
    >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.double)
    >>> x = W1[:, 0]
    >>> get_child(x, W2)
    [0, 1]
    """
    if x is not np.matrixlib.defmatrix.matrix:
        vec = np.mat(x).T
    else:
        vec = x
    if vec.shape[1]!=1:
        vec = vec.T
    h = decompose_with_dict(vec, dic)
    return list(filter(lambda x: h[x]>tol, range(len(h))))


class CommonSpace:
    """
    用来定义一个公共子空间的节点，包括节点分解情况
    """

    def __init__(self, speech_dic, noise_dic, cs_tol = 0.01):
        """
        用来构建一个包含子空间的信息实体

        Example:
        >>> W1 = np.mat([[0.5, 0, 0], [0.5, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=np.double)
        >>> W2 = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.double)
        >>> cs = CommonSpace(W1, W2)
        """
        self._speech_dic = speech_dic
        self._noise_dic = noise_dic
        self._speech_child_dic = {}
        self._speech_cs_set = set()
        self._noise_list = []
        self._learning_table = []
        for i in range(self._noise_dic.shape[1]):
            x = np.mat(self._noise_dic[:, i]).T
            if is_in_dict(x, self._speech_dic, tol = cs_tol):
                self._noise_list.append(i)
                children = get_child(x, self._speech_dic, tol=cs_tol)
                self._speech_child_dic[i] = children
                self._speech_cs_set.update(children)
        self._noise_list = sorted(self._noise_list)


    def fit_noise(self, noise_spec):
        """
        对已知的字典，使用字典分解，来学习公共字典对特征字典的对应关系

        Example:
        >>> import librosa
        >>> from utils import get_spec, norm_signal
        >>> noise, _ = librosa.load('data/noise/factory1.wav', sr=SAMPLE_RATE)
        >>> speech, _ = librosa.load('data/train/sa1.wav', sr=SAMPLE_RATE)
        >>> speech_spec = get_spec(norm_signal(speech))
        >>> noise_spec = get_spec(norm_signal(noise))
        >>> speech_dic, _ = decompose(np.abs(speech_spec), k=60)
        >>> noise_dic, _ = decompose(np.abs(noise_spec), k=20)
        >>> cs = CommonSpace(speech_dic, noise_dic)
        >>> cs.fit_noise(np.abs(noise_spec))
        """
        H = decompose_with_dict(noise_spec, self._noise_dic)
        for i in range(H.shape[1]):
            data = H[:, i]
            x = np.delete(data, self._noise_list)
            y = data[self._noise_list]
            self._learning_table.append((x, y))

    def get_total_dic(self):
        """
        获取最后的分解字典

        Example:
        >>> import librosa
        >>> from utils import get_spec, norm_signal
        >>> noise, _ = librosa.load('data/noise/factory1.wav', sr=SAMPLE_RATE)
        >>> speech, _ = librosa.load('data/train/sa1.wav', sr=SAMPLE_RATE)
        >>> speech_spec = get_spec(norm_signal(speech))
        >>> noise_spec = get_spec(norm_signal(noise))
        >>> speech_dic, _ = decompose(np.abs(speech_spec), k=60)
        >>> noise_dic, _ = decompose(np.abs(noise_spec), k=20)
        >>> cs = CommonSpace(speech_dic, noise_dic)
        >>> dic = cs.get_total_dic()
        """
        try:
            w_s = np.column_stack([self._speech_dic[:, index] for index in filter(lambda x:x not in self._speech_cs_set, range(self._speech_dic.shape[1]))])
        except:
            w_s = None
        try:
            w_n = np.column_stack([self._noise_dic[:, index] for index in filter(lambda x:x not in self._noise_list, range(self._noise_dic.shape[1]))])
        except:
            w_n = None
        self._n_len = w_n.shape[1]
        self._s_len = w_s.shape[1]
        try:
            indexs = sorted(list(self._speech_cs_set))
            w_cs = np.column_stack([self._speech_dic[:, index] for index in indexs])
            self._cs_len = w_cs.shape[1]
            self._restruct_nd = np.column_stack(filter(lambda x: x is not None, [w_cs, w_n]))
            return np.column_stack(filter(lambda x: x is not None, [w_s, w_cs, w_n]))
        except:
            self._cs_len = 0
            self._restruct_nd = np.column_stack(filter(lambda x: x is not None, [w_n]))
            return np.column_stack([w_s, w_n])


    def cos_distance(self,vector1, vector2):
        """
        计算向量见的余弦相似度

        Example:
        >>> v1 = [1, 0]
        >>> v2 = [0, 1]
        >>> CommonSpace.cos_distance(v1, v2)
        0.0
        """
        return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))

    def predict(self, h):
        """
        根据之前的noise激活信息，预测一个公共空间内的系数
        TODO: 返回扩增后的

        Example:
        >>> w_speech = np.mat([[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.double)
        >>> w_noise = np.mat([[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0], [0, 0, 0]], dtype=np.double)
        >>> cs = CommonSpace(w_speech, w_noise)
        >>> cs.fit_noise(np.mat([[0.4, 8.2, 2.9], [0.4, 8.2, 2.9], [0.4, 8.2, 2.9], [1,   2,   3], [2,   1,   3], [0,   0,   0]], dtype=np.double))
        >>> tmp = cs.get_total_dic()
        >>> cs.predict([0, 0, 0, 0, 4, 2])
        array([0.8])
        """
        h_n = h[-self._n_len:]
        max_err = -2
        max_index = -1
        for i, v in enumerate(self._learning_table):
            err = self.cos_distance(h_n, v[0])
            if err > max_err:
                max_err = err
                max_index = i
        cs_part = self._learning_table[max_index][1] * ((np.linalg.norm(h_n) / np.linalg.norm(self._learning_table[max_index][0])))
        return np.mat([int(i) for i in cs_part.reshape(-1, 1)] + [int(i) for i in h_n.reshape(-1, 1)]).T

    def predict_rate(self, h, rate):
        """
        根据之前的noise激活信息，预测一个公共空间内的系数
        TODO: 返回扩增后的

        Example:
        >>> w_speech = np.mat([[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 0],[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.double)
        >>> w_noise = np.mat([[0, 0, 1],[0, 0, 1],[0, 0, 1],[0, 1, 0],[1, 0, 0], [0, 0, 0]], dtype=np.double)
        >>> cs = CommonSpace(w_speech, w_noise)
        >>> cs.fit_noise(np.mat([[0.4, 8.2, 2.9], [0.4, 8.2, 2.9], [0.4, 8.2, 2.9], [1,   2,   3], [2,   1,   3], [0,   0,   0]], dtype=np.double))
        >>> tmp = cs.get_total_dic()
        >>> cs.predict([0, 0, 0, 0, 4, 2])
        array([0.8])
        """
        h_n = h[-self._n_len:]
        # print(h.shape)
        h_cs = h[self._s_len:self._s_len+self._cs_len]
        return np.mat([int(i) for i in h_cs.reshape(-1, 1)] + [int(i) for i in h_n.reshape(-1, 1)]).T


    def restruct_noise_dic(self):
        """
        返回重构用的噪声字典
        """
        return self._restruct_nd
        w_n = np.column_stack([self._noise_dic[:, index] for index in filter(lambda x:x not in self._noise_list, range(self._noise_dic.shape[1]))])
        try:
            w_cs = np.column_stack([self._noise_dic[:, index] for index in self._noise_list])
            return np.column_stack([w_cs, w_n])
        except:
            return w_n

if __name__=='__main__':
    import doctest
    doctest.testmod()

