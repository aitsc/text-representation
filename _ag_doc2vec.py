from tqdm import tqdm
import gensim
from gensim.models import Doc2Vec
import numpy as np
import os
import time
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class Doc2vec:
    def __init__(self, dataIRaddress, vector_size=200, window=8, min_count=1, workers=4, epochs=10, 模型保存地址=None, 模型读取地址=None):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(dataIRaddress)
        self._trainSenVec, self._testSenVec = self._train(vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs, 模型保存地址=模型保存地址, 模型读取地址=模型读取地址)

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

    def _train(self, vector_size=200, window=8, min_count=1, workers=4, epochs=10, 模型保存地址=None, 模型读取地址=None):
        if not 模型读取地址 or not os.path.exists(模型读取地址):
            documents = []
            for paperID, line in self._trainPaper_sentence_L:
                documents.append(gensim.models.doc2vec.TaggedDocument(line.split(), [paperID]))
            for paperID, line in self._testPaper_sentence_L:
                documents.append(gensim.models.doc2vec.TaggedDocument(line.split(), [paperID]))
            print('开始训练doc2vec...')
            model = Doc2Vec(documents, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)
            if 模型保存地址:
                print('保存模型...')
                model.save(模型保存地址)
        else:
            print('开始读取doc2vec模型...')
            model = Doc2Vec.load(模型读取地址)

        trainSenVec, textSenVec = [], []
        for paperID, _ in self._trainPaper_sentence_L:
            trainSenVec.append(model.docvecs[paperID])
        for paperID, _ in self._testPaper_sentence_L:
            textSenVec.append(model.docvecs[paperID])
        return np.array(trainSenVec), np.array(textSenVec)

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

    def 计算分类数据集相似度矩阵(self):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        print('cos计算中...')
        # 每一行表示一个训练集论文和所有测试集论文的相似度
        test_train_dis_L = -np.dot(self._testSenVec, self._trainSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._trainSenVec, axis=1), axis=0))

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
    数据集地址 = 'data/TC AGnews/dataset.text'
    startTime = time.time()

    Doc2vec_obj = Doc2vec(dataIRaddress=数据集地址,
                          vector_size=100, window=8, min_count=1, workers=10, epochs=50,
                          # 模型保存地址='data/IR arxiv/ag_doc2vec.model',
                          # 模型读取地址='data/IR arxiv/ag_doc2vec.model',
                          )

    # 检索数据集评估
    # test_train_D = Doc2vec_obj.计算测试集与训练集余弦相似度()
    # IR评估_obj = IR评估(标签地址=数据集地址)
    # IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    test_train_dis_L, trainLabel_L, testLabel_L = Doc2vec_obj.计算分类数据集相似度矩阵()
    TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
                n_neighbors=[3, 6, 9, 12, 15, 18],
                n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
