from gensim.summarization.bm25 import BM25  # gensim==3.7.1
from multiprocessing import Pool
from tqdm import tqdm
import time
from functools import partial
import sys
import math
import pickle
import numpy as np
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


def _get_scores(bm25, document):  # 放在这, 防止所有数据都被进程复制一份
    return bm25.get_scores(document)


class OkapiBM25:
    def __init__(self, 数据集地址):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(数据集地址)
        print('BM25模型初始化...')
        self._bm25 = BM25([i.split() for _, i in self._trainPaper_sentence_L] + [i.split() for _, i in self._testPaper_sentence_L])

    @staticmethod
    def _getPaperInfor(dataIRaddress):
        trainPaper_sentence_L = []  # [(训练集论文,句子),..]
        testPaper_sentence_L = []  # [(测试集论文,句子),..]
        testPaper_IRpaper_D = {}  # {测试集论文:训练集论文set,..}

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取论文信息'):
                if i == 0:
                    testPaper_IRpaper_D = eval(line.strip())
                    continue
                line = line.split('\t')
                paperID = line[0]
                if paperID in testPaper_IRpaper_D:
                    testPaper_sentence_L.append((paperID, 句子清洗(line[1] + ' ' + line[2])))
                else:
                    trainPaper_sentence_L.append((paperID, 句子清洗(line[1] + ' ' + line[2])))
        return trainPaper_sentence_L, testPaper_sentence_L

    def 计算测试集与训练集相似度(self, workers):
        print('测试集与训练集相似度计算中...')
        get_score = partial(_get_scores, self._bm25)
        pool = Pool(workers)
        test_paper_sim_L = pool.map(get_score, [i.split() for _, i in self._testPaper_sentence_L])
        pool.close()
        pool.join()

        # 还原编号
        test_train8sim_D = {}  # {测试集论文编号:{训练集论文编号:相似度,..},..}
        for i in tqdm(range(len(test_paper_sim_L)), '还原编号, 测试集数'):
            testID = self._testPaper_sentence_L[i][0]
            d = {}
            for j, sim in enumerate(test_paper_sim_L[i][:len(self._trainPaper_sentence_L)]):
                trainID = self._trainPaper_sentence_L[j][0]
                d[trainID] = sim
            test_train8sim_D[testID] = d
        # 排序
        for k in tqdm(test_train8sim_D.keys(), '排序'):
            test_train8sim_D[k] = sorted(test_train8sim_D[k].items(), key=lambda t: t[1], reverse=True)
        self._test_train8sim_D = test_train8sim_D  # {测试集论文编号:[(训练集论文编号,相似度),..],..}
        test_train_D = {k: [i[0] for i in v] for k, v in test_train8sim_D.items()}
        return test_train_D  # {测试集论文编号:[训练集论文编号,..]}

    @staticmethod
    def _compute_TtoT_prob_singleProcess(args):
        paper_sentence_L, noL, topK, bm25 = args  # noL 表示训练论文在 allIdL 中的序号

        输出千分比 = 1
        概率矩阵 = []  # [(paperID,[prob,..],[no,..]),..], no表示在idL中的序号
        for i, (paperID, sen) in enumerate(paper_sentence_L):
            permillage = int(i / len(paper_sentence_L) * 1000)
            if permillage > 输出千分比:
                输出千分比 = permillage
                sys.stdout.write('\r')
                print('%.1f%%---' % (permillage / 10,), end='')
                sys.stdout.flush()

            result = bm25.get_scores(sen.split())
            result[noL[i]] = 0  # 自己与自己的相似度置零
            no_sim_L = [(i, j) for i, j in enumerate(result)]  # 准备排序
            no_sim_L = sorted(no_sim_L, key=lambda t: t[1], reverse=True)[:topK]  # 相似度越大越相似, 取topK

            s = sum([i for _, i in no_sim_L])  # 用于归一化
            if s == 0:  # 如果没有相似的
                prob = [1 / topK] * topK
            else:
                prob = [i / s for _, i in no_sim_L]
            paperID_probL_noL = (paperID, prob, [i for i, _ in no_sim_L])  # 构建三元组结果
            概率矩阵.append(paperID_probL_noL)
        sys.stdout.write('\r')
        return 概率矩阵

    def compute_TtoT_prob(self, 进程数=2, 存储地址='', 含测试文本=True, topK=0.01):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        paper_sentence_L = self._trainPaper_sentence_L
        if 含测试文本:
            paper_sentence_L += self._testPaper_sentence_L
        if 0 < topK < 1:
            topK = round(topK * len(paper_sentence_L))
        elif topK <= 0:
            topK = len(paper_sentence_L)

        avgText = math.ceil(len(paper_sentence_L) / 进程数)
        参数l = [(paper_sentence_L[avgText * i: avgText * (i+1)], list(range(avgText * i, avgText * (i+1))), topK, self._bm25) for i in range(进程数)]
        pool = Pool(进程数)
        paperID_probL_noL_L_L = pool.map(self._compute_TtoT_prob_singleProcess, 参数l)
        pool.close()
        pool.join()

        paperID_probL_noL_L = sum(paperID_probL_noL_L_L, [])  # [(paperID,[prob,..],[no,..]),..], no表示在idL中的序号
        # for i in range(0, 50):  # 测试, 查看是否和为1
        #     print(sum(paperID_probL_noL_L[i][1]))
        #     print(min(paperID_probL_noL_L[i][1]))
        #     print(max(paperID_probL_noL_L[i][1]))
        #     print()
        # 保存
        if 存储地址:
            二进制 = pickle.dumps(paperID_probL_noL_L)
            缓存 = 10 ** 6
            with open(存储地址.encode('utf-8'), 'wb') as w:
                for i in tqdm(range(0, len(二进制), 缓存), '写入 paperID_probL_idL_L'):
                    w.write(二进制[i:i + 缓存])

    def 计算分类数据集相似度矩阵(self, workers):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        print('cos计算中...')
        # 每一行表示一个训练集论文和所有测试集论文的相似度
        print('测试集与训练集相似度计算中...')
        bm25 = BM25([i.split() for _, i in self._trainPaper_sentence_L])
        get_score = partial(_get_scores, bm25)
        pool = Pool(workers)
        test_paper_sim_L = pool.map(get_score, [i.split() for _, i in self._testPaper_sentence_L])
        pool.close()
        pool.join()
        test_train_dis_L = -np.array(test_paper_sim_L)  # 相似度越高距离越小

        # 获取标签
        trainLabel_L = []
        testLabel_L = []
        for textID, _ in self._trainPaper_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            trainLabel_L.append(label)
        for textID, _ in self._testPaper_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            testLabel_L.append(label)
        return test_train_dis_L, trainLabel_L, testLabel_L


if __name__ == '__main__':
    数据集地址 = 'data/CR dblp/dataset.text'
    startTime = time.time()

    OkapiBM25_obj = OkapiBM25(数据集地址)
    # OkapiBM25_obj.compute_TtoT_prob(进程数=8,
    #                                 存储地址='data/IR arxiv/ai_allTtoT100-paperID_probL_noL_L.pkl',
    #                                 含测试文本=True,
    #                                 topK=100)

    # 检索数据集评估
    test_train_D = OkapiBM25_obj.计算测试集与训练集相似度(workers=4)
    IR评估_obj = IR评估(标签地址=数据集地址)
    IR评估_obj.评估(预测标签=test_train_D, topN=1000, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_train_dis_L, trainLabel_L, testLabel_L = OkapiBM25_obj.计算分类数据集相似度矩阵(workers=4)
    # TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=4, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
