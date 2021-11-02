from tqdm import tqdm
import numpy as np
import time
import random
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class EmbaddingAvg:
    def __init__(self, 数据集地址, 词向量地址=None, 句向量=None):
        '''
        :param 数据集地址:
        :param 词向量地址:
        :param 句向量: int表示随机句向量; str表示有句向量文件,句向量文件和数据集每一条对应,空格分割,没有多余的描述,比如Doc2VecC
        '''
        print('数据集: %s' % 数据集地址)
        self._trainPaper_sentence_L, self._testPaper_sentence_L, self._textNo_isTrain_L = self._getPaperInfor(数据集地址)
        if isinstance(句向量, int):
            print('随机获取句向量...')
            self._trainSenVec, self._testSenVec = self._getRandomSenEmbedding(self._trainPaper_sentence_L, self._testPaper_sentence_L, 句向量)
        elif isinstance(句向量, str) and 句向量:
            print('提取句向量文件...')
            self._trainSenVec, self._testSenVec = [], []
            with open(句向量, 'r', encoding='utf-8') as r:
                for i, line in enumerate(r):
                    vec = [float(i) for i in line.strip().split(' ')]
                    if i < len(self._textNo_isTrain_L):
                        if self._textNo_isTrain_L[i]:
                            self._trainSenVec.append(vec)
                        else:
                            self._testSenVec.append(vec)
                    else:
                        break
            self._trainSenVec, self._testSenVec = np.array(self._trainSenVec), np.array(self._testSenVec)
        else:
            print('获取句向量...')
            self._trainSenVec, self._testSenVec = self._getAvgEmbedding(词向量地址, self._trainPaper_sentence_L, self._testPaper_sentence_L)
        print('trainSenVec dim: %s, testSenVec dim: %s' % (str(self._trainSenVec.shape), str(self._testSenVec.shape)))

    @staticmethod
    def _getPaperInfor(dataIRaddress):
        trainPaper_sentence_L = []  # [(训练集论文,句子),..]
        testPaper_sentence_L = []  # [(测试集论文,句子),..]
        testPaper_IRpaper_D = {}  # {测试集论文:训练集论文set,..}
        textNo_isTrain_L = []  # [isTrain,..], no从0开始, 表示这个句子是不是训练集, 用于句向量文件的匹配

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取论文信息'):
                if i == 0:
                    testPaper_IRpaper_D = eval(line.strip())
                    continue
                line = line.split('\t')
                paperID = line[0]
                if paperID in testPaper_IRpaper_D:
                    testPaper_sentence_L.append((paperID, 句子清洗(line[1] + ' ' + line[2])))
                    textNo_isTrain_L.append(False)
                else:
                    trainPaper_sentence_L.append((paperID, 句子清洗(line[1] + ' ' + line[2])))
                    textNo_isTrain_L.append(True)
        return trainPaper_sentence_L, testPaper_sentence_L, textNo_isTrain_L

    @staticmethod
    def _getAvgEmbedding(词向量地址, trainPaper_sentence_L, testPaper_sentence_L):
        trainSenVec = []
        testSenVec = []
        all_words = set()  # 获得所有词集合
        for _, i in trainPaper_sentence_L + testPaper_sentence_L:
            all_words |= set(i.split())
        # 读取所有词向量
        词_向量d = {}
        维度 = -1
        if 词向量地址:
            with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8', errors='ignore') as r:
                for line in tqdm(r, '读取词向量'):
                    line = line.strip().split(' ')
                    if len(line) < 3:
                        continue
                    word = line[0]
                    if word in all_words:
                        词_向量d[word] = np.array([float(i) for i in line[1:]])
                        if 维度 <= 0:
                            维度 = len(词_向量d[word])
        else:  # 随机词向量
            维度 = 100
            for w in tqdm(all_words, '构建随机词向量'):
                词_向量d[w] = [random.random() for _ in range(维度)]
        # 计算平均词向量
        for _, sen in tqdm(trainPaper_sentence_L, '计算训练集平均词向量'):
            wordsL = sen.split()
            x = np.array([0.]*维度)
            n = 0
            for word in wordsL:
                if word in 词_向量d:
                    x += 词_向量d[word]
                    n += 1
            trainSenVec.append(x / n)
        for _, sen in tqdm(testPaper_sentence_L, '计算测试集平均词向量'):
            wordsL = sen.split()
            x = np.array([0.]*维度)
            n = 0
            for word in wordsL:
                if word in 词_向量d:
                    x += 词_向量d[word]
                    n += 1
            testSenVec.append(x / n)
        return np.array(trainSenVec), np.array(testSenVec)

    @staticmethod
    def _getRandomSenEmbedding(trainPaper_sentence_L, testPaper_sentence_L, dim = 100):
        trainSenVec = [[random.random() for _ in range(dim)] for _ in range(len(trainPaper_sentence_L))]
        testSenVec = [[random.random() for _ in range(dim)] for _ in range(len(testPaper_sentence_L))]
        return np.array(trainSenVec), np.array(testSenVec)

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
    数据集地址 = 'data/CR dblp/dataset.text'
    startTime = time.time()

    EmbaddingAvg_obj = EmbaddingAvg(数据集地址=数据集地址,
                                    # 词向量地址='data/TC AGnews/ak_Corpus_vectors.text',  # 如果没有地址则随机词向量
                                    # 词向量地址='data/IR arxiv/ak_glove-Corpus_vectors.text',
                                    # 词向量地址='data/CR dblp/ak_doc2vecc_wvs.text',
                                    # 句向量='data/CR dblp/ak_doc2vecc_dvs.text',  # 如果是一个数则使用随机句向量, 优先于词向量
                                    句向量='data/CR dblp/bg_doc_vectors.text',
                                    )

    # 检索数据集评估
    test_train_D = EmbaddingAvg_obj.计算测试集与训练集余弦相似度()
    IR评估_obj = IR评估(标签地址=数据集地址)
    IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_train_dis_L, trainLabel_L, testLabel_L = EmbaddingAvg_obj.计算分类数据集相似度矩阵()
    # TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
