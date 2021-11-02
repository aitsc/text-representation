import numpy as np
import math
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys


class IR评估:
    def __init__(self, 标签地址):
        self._编号_检索论文d = self._载入标准评价(标签地址)  # {测试集论文编号:[检索论文,..],..}

    @staticmethod
    def _载入标准评价(标准评价地址):
        编号_检索论文d = {}
        with open(标准评价地址.encode('utf-8'), 'r', encoding='utf-8') as r:
            for i, line in enumerate(r):
                if i == 0 and line[0] == '{':
                    编号_检索论文d = eval(line.strip())
                    for k in 编号_检索论文d.keys():
                        编号_检索论文d[k] = list(编号_检索论文d[k])
                    break
                line = line.strip().split('\t')
                编号_检索论文d[line[0]] = line[1:]
        return 编号_检索论文d

    def 评估(self, 预测标签, topN=20, 简化=True, 输出地址=None, 输出控制台=True):
        '''
        :param 预测标签:
        :param topN:
        :param 简化:
        :param 输出地址:
        :param 输出控制台:
        :return:
        '''
        预测向量组l, 标签向量组l = [], []
        论文编号l = []
        for 编号, 检索论文l in self._编号_检索论文d.items():
            标签向量组l.append(检索论文l)
            预测向量组l.append(预测标签[编号])
            论文编号l.append(编号)

        if 简化:
            top遍历 = [topN]
        else:
            top遍历 = range(1, topN+1)

        batch_top_FP, batch_top_FN, batch_top_TP = [], [], []
        for 预测向量组, 标签向量组 in tqdm(zip(预测向量组l, 标签向量组l), '评估'):
            top_FP, top_FN, top_TP = [], [], []

            标签向量组s = set(标签向量组)
            正例数 = len(标签向量组s)
            for j in top遍历:
                预测正确数 = len(set(预测向量组[:j]) & 标签向量组s)
                top_FP.append(j-预测正确数)
                top_FN.append(正例数-预测正确数)
                top_TP.append(预测正确数)
            batch_top_FP.append(top_FP)
            batch_top_FN.append(top_FN)
            batch_top_TP.append(top_TP)

        MAP_l, NDCG_l, bpref_l = [], [], []
        for j in top遍历:
            MAP_l.append(self.MAP_相关文档数为N(预测向量组l, 标签向量组l, j))
            NDCG_l.append(self.NDCG_无序(预测向量组l, 标签向量组l, j))
            bpref_l.append(self.Bpref_相关文档数为N(预测向量组l, 标签向量组l, j))

        batch_top_FP = np.array(batch_top_FP)
        batch_top_FN = np.array(batch_top_FN)
        batch_top_TP = np.array(batch_top_TP)
        # 所有
        batch_top_P = batch_top_TP/(batch_top_TP+batch_top_FP)
        batch_top_R = batch_top_TP/(batch_top_TP+batch_top_FN)
        # 整体
        top_macroP = batch_top_P.mean(axis=0)
        top_macroR = batch_top_R.mean(axis=0)
        top_macroF1 = 2*top_macroP*top_macroR/(top_macroP+top_macroR)

        if 简化:
            top_macroP = float(top_macroP[0])
            top_macroR = float(top_macroR[0])
            top_macroF1 = float(top_macroF1[0])
            MAP = MAP_l[0]
            NDCG = NDCG_l[0]
            bpref = bpref_l[0]
        else:
            top_macroP = top_macroP.tolist()
            top_macroR = top_macroR.tolist()
            top_macroF1 = top_macroF1.tolist()
            MAP = MAP_l
            NDCG = NDCG_l
            bpref = bpref_l
        输出 = {
            'macro-P': top_macroP,
            'macro-R': top_macroR,
            'macro-F1': top_macroF1,
            'MAP': MAP,
            'NDCG': NDCG,
            'bpref': bpref,
        }
        if 输出控制台:
            print('指标:\tmacro-P\tmacro-R\tmacro-F1\tMAP\tNDCG\tbpref')
            print('结果:\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (top_macroP, top_macroR, top_macroF1, MAP, NDCG, bpref))
        if 输出地址:
            with open(输出地址.encode('utf-8'), 'w', encoding='utf-8') as w:
                w.write('all: '+str(输出)+'\n')
                w.write('manuscript\tP\tR\n')
                for 编号, top_P, top_R in zip(论文编号l, batch_top_P, batch_top_R):
                    top_P, top_R = top_P.tolist(), top_R.tolist()
                    w.write(编号+'\t')
                    if 简化:
                        w.write('%.4f\t%.4f\n' % (top_P[-1], top_R[-1]))
                    else:
                        w.write('%s\t%s\n' % (str(top_P), str(top_R)))
        return 输出

    @staticmethod
    def MAP_相关文档数为N(预测向量组l, 标签向量组l, N):
        AP = []
        相关文档数 = N
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            准确个数 = 0
            ap = 0
            for j in range(相关文档数):
                if 预测向量l[j] in 标签向量s:
                    准确个数 += 1
                    ap += 准确个数 / (j + 1)
            AP.append(ap / 相关文档数)
        return sum(AP) / len(AP)

    @staticmethod
    def NDCG_无序(预测向量组l, 标签向量组l, N):
        ndcg = 0
        idcg = sum(1 / math.log2(i + 2) for i in range(N))
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            for j in range(len(预测向量l)):
                if 预测向量l[j] in 标签向量s:
                    ndcg += 1 / math.log2(j + 2)
        return ndcg / idcg / len(预测向量组l)

    @staticmethod
    def Bpref_相关文档数为N(预测向量组l, 标签向量组l, N):
        bpref = 0
        相关文档数 = N
        for i in range(len(预测向量组l)):
            预测向量l = 预测向量组l[i][:N]
            标签向量s = set(标签向量组l[i])
            不准确个数 = 0
            for j in range(相关文档数):
                if 预测向量l[j] not in 标签向量s:
                    不准确个数 += 1
                bpref += (1 - 不准确个数 / 相关文档数)
        return bpref / 相关文档数 / len(预测向量组l)


class TC评估:
    @staticmethod
    def cos_dis(x, y):
        return -float(cosine_similarity([x], [y]))

    @staticmethod
    def coupled_dis(x, y):
        length = int(len(x)/2)
        dis1 = -float(cosine_similarity([x[:length]], [y[length:]]))
        dis2 = -float(cosine_similarity([x[length:]], [y[:length]]))
        return dis1 + dis2

    @staticmethod
    def 评估(trainVec_L, trainLabel_L, testVec_L, testLabel_L, n_neighbors, n_jobs, metric, 输出控制台=True):
        '''
        :param trainVec_L:
        :param trainLabel_L:
        :param testVec_L:
        :param testLabel_L:
        :param n_neighbors:
        :param n_jobs:
        :param metric: 距离越小越相似
        :param 输出控制台:
        :return:
        '''
        neigh = kNN(n_neighbors=n_neighbors, n_jobs=n_jobs, metric=metric)
        print('kNN训练...')
        neigh.fit(trainVec_L, trainLabel_L)
        print('计算acc...')
        acc = neigh.score(testVec_L, testLabel_L)
        if 输出控制台:
            print('指标:\tacc\terror rate')
            print('结果:\t%.4f\t%.4f' % (acc, 1-acc))
        输出 = {
            'acc': acc,
            'error rate': 1-acc,
        }
        return 输出

    @staticmethod
    def 距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L, n_neighbors, n_jobs, knn使用距离加权=False, 输出控制台=True):
        '''
        距离越小越相似
        :param test_train_dis_L: 每一行是一个测试样本和所有训练样本的距离
        :param trainLabel_L:
        :param testLabel_L:
        :param n_neighbors: int or list
        :param n_jobs:
        :param knn使用距离加权:
        :param 输出控制台:
        :return:
        '''
        if knn使用距离加权:
            weights = 'distance'
        else:
            weights = 'uniform'
        if isinstance(n_neighbors, int) or len(n_neighbors) == 1:
            if isinstance(n_neighbors, list):
                n_neighbors = n_neighbors[0]
            print('开始计算kNN分类结果...', end='')
            sys.stdout.flush()
            startTime = time.time()
            neigh = kNN(n_neighbors=n_neighbors, n_jobs=n_jobs, algorithm='brute', weights=weights,
                        metric=lambda x, y, m: m[int(x)][int(y)], metric_params={'m': test_train_dis_L})
            neigh.fit([[j] for j in range(len(trainLabel_L))], trainLabel_L)
            acc = neigh.score([[j] for j in range(len(testLabel_L))], testLabel_L)
            print('%.2fm' % ((time.time() - startTime) / 60))
            if 输出控制台:
                print('指标:\tacc\terror rate')
                print('结果:\t%.4f\t%.4f' % (acc, 1-acc))
            acc_L = [acc]
        else:
            acc_L = []
            for k in tqdm(n_neighbors, '计算kNN, k='):
                neigh = kNN(n_neighbors=k, n_jobs=n_jobs, algorithm='brute', weights=weights,
                            metric=lambda x, y, m: m[int(x)][int(y)], metric_params={'m': test_train_dis_L})
                neigh.fit([[j] for j in range(len(trainLabel_L))], trainLabel_L)
                acc = neigh.score([[j] for j in range(len(testLabel_L))], testLabel_L)
                acc_L.append(acc)
            if 输出控制台:
                print('k:\t'+'\t'.join([str(i) for i in n_neighbors]))
                print('acc:\t'+'\t'.join(['%.4f' % i for i in acc_L]))
                print('error rate:\t'+'\t'.join(['%.4f' % (1-i) for i in acc_L]))
        输出 = {
            'acc': acc_L,
            'error rate': [1-i for i in acc_L],
        }
        return 输出
