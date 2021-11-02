# 2020-06-05 从 shiyan 到 text representation
import sys
import h5py
import numpy as np
import os
import time
from tqdm import tqdm


if __name__ == '__main__':
    path = 'av_xlnet-large-cased_sen_mat_len'
    print(path)
    h5f_r = h5py.File(r'D:\data\code\python\paper\text representation\data\CR dblp/' + path + '.h5', 'r')
    h5f_w = h5py.File(r'F:\data\_large_tem_file\CTE\data\CR dblp/' + path + '.h5', 'w')

    句子长度, 词维度 = h5f_r['matrixs'].shape[1], h5f_r['matrixs'].shape[2]
    总数 = h5f_r['matrixs'].shape[0]
    h5f_w.create_dataset("sentences", (总数,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
    h5f_w.create_dataset("matrixs", (总数, 句子长度, 词维度), maxshape=(None, 句子长度, 词维度), dtype=np.float32, chunks=(1, 句子长度, 词维度))
    h5f_w.create_dataset("lengths", (总数,), maxshape=(None,), dtype=np.int32, chunks=(1,))

    for i in tqdm(range(0, 总数, 100)):
        h5f_w['sentences'][i: min(i + 100, 总数)] = h5f_r['sentences'][i: min(i+100, 总数)]
        h5f_w['matrixs'][i: min(i + 100, 总数)] = h5f_r['matrixs'][i: min(i+100, 总数)]
        h5f_w['lengths'][i: min(i + 100, 总数)] = h5f_r['lengths'][i: min(i+100, 总数)]
        h5f_w.flush()
    h5f_w.close()
    h5f_r.close()
