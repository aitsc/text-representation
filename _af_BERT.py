from bert_serving.client import BertClient
from tqdm import tqdm
import numpy as np
import time
import sys
import pickle
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class BERT:
    def __init__(self, 数据集地址):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(数据集地址)
        self._BertClient = BertClient(port=5560, port_out=5561)  # 保证客户端是获取句向量
        print('获取句向量...')
        self._trainSenVec = self._BertClient.encode([i for _, i in self._trainPaper_sentence_L])
        self._testSenVec = self._BertClient.encode([i for _, i in self._testPaper_sentence_L])

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

    def 计算测试集与训练集余弦相似度(self):
        print('cos计算中...')
        # 每一行表示一个训练集论文和所有测试集论文的相似度
        train_test_sim_L = (np.dot(self._testSenVec, self._trainSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._trainSenVec, axis=1), axis=0))).T

        # 还原编号
        test_train8sim_D = {}  # {测试集论文编号:{训练集论文编号:相似度,..},..}
        for i in tqdm(range(len(train_test_sim_L)), '还原编号, 训练集数'):
            for j, sim in enumerate(train_test_sim_L[i]):
                testID = self._testPaper_sentence_L[j][0]
                trainID = self._trainPaper_sentence_L[i][0]
                if testID in test_train8sim_D:
                    test_train8sim_D[testID][trainID] = sim
                else:
                    test_train8sim_D[testID] = {trainID: sim}
        # 排序
        for k in tqdm(test_train8sim_D.keys(), '排序'):
            test_train8sim_D[k] = sorted(test_train8sim_D[k].items(), key=lambda t: t[1], reverse=True)
        self._test_train8sim_D = test_train8sim_D  # {测试集论文编号:[(训练集论文编号,相似度),..],..}
        test_train_D = {k: [i[0] for i in v] for k, v in test_train8sim_D.items()}
        return test_train_D  # {测试集论文编号:[训练集论文编号,..]}

    def 计算所有论文之间余弦相似度(self, 归一化=True, 自我相似度置零=True, 矩阵存储地址='', 坐标编号存储地址=''):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        startTime = time.time()

        textSenVec = np.vstack((self._trainSenVec, self._testSenVec))
        paperToPaper_sim_M = np.dot(textSenVec, textSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(textSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(textSenVec, axis=1), axis=0))
        if 归一化:
            paperToPaper_sim_M = 100 ** paperToPaper_sim_M  # 防止负数，并拉开相似度差距
            if 自我相似度置零:
                for i in range(len(paperToPaper_sim_M)):
                    paperToPaper_sim_M[i, i] = 0.  # 自己选择自己的概率为0
            paperToPaper_sim_M = paperToPaper_sim_M / paperToPaper_sim_M.sum(1)  # 归一化
        for i in range(0, 50):
            print(sum(paperToPaper_sim_M[i]))
        print(paperToPaper_sim_M.min(), paperToPaper_sim_M.max())
        paperToPaper_sim_L = paperToPaper_sim_M.tolist()
        print('%.2fm' % ((time.time() - startTime) / 60))

        # 保存
        if 矩阵存储地址:
            二进制 = pickle.dumps(paperToPaper_sim_L)
            缓存 = 10 ** 6
            with open(矩阵存储地址.encode('utf-8'), 'wb') as w:
                for i in tqdm(range(0, len(二进制), 缓存), '写入相似度概率矩阵'):
                    w.write(二进制[i:i + 缓存])
        if 坐标编号存储地址:
            id_L = []
            for i, _ in self._trainPaper_sentence_L:
                id_L.append(i)
            for i, _ in self._testPaper_sentence_L:
                id_L.append(i)
            with open(坐标编号存储地址.encode('utf-8'), 'w', encoding='utf-8') as w:
                w.write(str(id_L))
        return paperToPaper_sim_L


if __name__ == '__main__':
    数据集地址 = 'data/IR arxiv/IR arxiv.text'
    startTime = time.time()

    BERT_obj = BERT(数据集地址=数据集地址)
    BERT_obj.计算所有论文之间余弦相似度(归一化=True, 自我相似度置零=True,
                             矩阵存储地址='data/IR arxiv/af_IR arxiv所有论文BERT距离.pkl',
                             坐标编号存储地址='data/IR arxiv/af_IR arxiv所有论文BERT距离-编号.text')
    # test_train_D = BERT_obj.计算测试集与训练集余弦相似度()
    #
    # IR评估_obj = IR评估(标签地址=数据集地址)
    # IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)
    print('%.2fm' % ((time.time() - startTime) / 60))


r'''
使用BERT前打开服务,比如执行命令: 
bert-serving-start -model_dir /home/tansc/python/fork/uncased_L-24_H-1024_A-16 -gpu_memory_fraction=0.5 -max_seq_len=200 -max_batch_size=256 -num_worker=1 -port 5560 -port_out 5561
bert-serving-start -model_dir D:\data\code\python\GPU-31-11\fork\uncased_L-24_H-1024_A-16 -gpu_memory_fraction=0.7 -max_seq_len=200 -max_batch_size=256 -num_worker=1 -port 5560 -port_out 5561
'''