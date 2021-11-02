from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing
from multiprocessing import Pool
import time
import sys
from tqdm import tqdm
import numpy as np
import math
import pickle
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
TC评估 = importlib.import_module('_ad_evaluate').TC评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗

class TF_IDF:
    def __init__(self, dataIRaddress):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(dataIRaddress)
        self._paperSen_csr_matrix = self._computeTfIdf(self._trainPaper_sentence_L + self._testPaper_sentence_L)  # 前面是训练集, 后面是测试集
        self._trainPaperSen_csr_matrix = self._paperSen_csr_matrix[:len(self._trainPaper_sentence_L)]  # 训练集和测试集整体计算得到的训练集
        self._testPaperSen_csr_matrix = self._paperSen_csr_matrix[len(self._trainPaper_sentence_L):]  # 训练集和测试集整体计算得到的测试集

    @staticmethod
    def _getPaperInfor(dataIRaddress):
        trainPaper_sentence_L = []  # [(训练集论文,句子),..]
        testPaper_sentence_L = []  # [(测试集论文,句子),..]
        testPaper_IRpaper_D = {}  # {测试集论文:训练集论文set,..}

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取文本信息'):
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

    def _computeTfIdf(self, paper_sentence_L):
        tfidf = TfidfVectorizer(token_pattern=r"\S+", stop_words='english')
        paperSenL = [j for i, j in paper_sentence_L]
        paperSen_csr_matrix = tfidf.fit_transform(paperSenL)
        return paperSen_csr_matrix

    @staticmethod
    def _cos_similarity(csr_matrix1, csr_matrix2, queue: multiprocessing.Queue, ID, minV=False):
        result = cosine_similarity(csr_matrix1, csr_matrix2, dense_output=False)
        if minV:
            minV = min(result.data)
        else:
            minV = None
        result = result.todense()
        queue.put((result, ID, minV))

    def 计算测试集与训练集余弦相似度(self, 进程数=1):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        avgPaper = math.ceil(len(self._trainPaper_sentence_L)/进程数)
        queue = multiprocessing.Queue()
        已开启进程数 = 0
        for ID in range(进程数):
            multiprocessing.Process(target=self._cos_similarity, args=(self._trainPaperSen_csr_matrix[avgPaper*ID: avgPaper*(ID+1)], self._testPaperSen_csr_matrix, queue, ID)).start()
            已开启进程数 += 1
        id_result_D = {}
        for i in tqdm(range(已开启进程数), '进程运算中'):
            result, ID, _ = queue.get()
            id_result_D[ID] = result.tolist()
        train_test_sim_L = []  # 每一行表示一个训练集论文和所有测试集论文的相似度
        for i in range(len(id_result_D)):
            train_test_sim_L += id_result_D[i]

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
            test_train8sim_D[k] = sorted(test_train8sim_D[k].items(), key=lambda t: t[1], reverse=True)  # {测试集论文编号:[(训练集论文编号,相似度),..],..}
        test_train_D = {k: [i[0] for i in v] for k, v in test_train8sim_D.items()}
        return test_train_D  # {测试集论文编号:[训练集论文编号,..]}

    def 计算所有论文之间余弦相似度(self, 进程数=1, 归一化=True, 自我相似度置零=True, 矩阵存储地址='', 坐标编号存储地址=''):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        allLen = len(self._trainPaper_sentence_L) + len(self._testPaper_sentence_L)
        avgPaper = math.ceil(allLen / 进程数)
        queue = multiprocessing.Queue()
        已开启进程数 = 0
        最小值 = float('inf')

        startTime = time.time()
        for ID in range(进程数):
            multiprocessing.Process(target=self._cos_similarity, args=(self._paperSen_csr_matrix[avgPaper*ID: avgPaper*(ID+1)], self._paperSen_csr_matrix, queue, ID, True)).start()
            已开启进程数 += 1
        id_result_D = {}
        for i in tqdm(range(已开启进程数), '进程运算中'):
            result, ID, minV = queue.get()
            if 归一化:
                最小值 = min(minV, 最小值)
            id_result_D[ID] = result
        paperToPaper_sim_M = np.vstack([i[1] for i in sorted(id_result_D.items())])  # numpy.matrixlib.defmatrix.matrix
        if 归一化:
            paperToPaper_sim_M += 最小值 / allLen  # 防止归一化时被除数为0
            if 自我相似度置零:
                for i in range(len(paperToPaper_sim_M)):
                    paperToPaper_sim_M[i, i] = 0.  # 自己选择自己的概率为0
            paperToPaper_sim_M = paperToPaper_sim_M / paperToPaper_sim_M.sum(1)  # 归一化
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
            with open(坐标编号存储地址, 'w', encoding='utf-8') as w:
                w.write(str(id_L))
        return paperToPaper_sim_L

    @staticmethod
    def _compute_TtoT_prob_singleProcess(args):
        paperSen_csr_matrix, idL, topK, indexL, indexR = args

        输出千分比 = 1
        概率矩阵 = []  # [(paperID,[prob,..],[no,..]),..], no表示在idL中的序号
        for i, index in enumerate(range(indexL, min(indexR, len(idL)))):
            permillage = int(i / (indexR - indexL) * 1000)
            if permillage > 输出千分比:
                输出千分比 = permillage
                sys.stdout.write('\r')
                print('%.1f%%---' % (permillage / 10,), end='')
                sys.stdout.flush()

            result = cosine_similarity(paperSen_csr_matrix[index: index+1], paperSen_csr_matrix, dense_output=False)
            result = result.todense().tolist()[0]  # 一个相似度向量
            result[index] = 0  # 自己与自己的相似度置零
            no_sim_L = [(i, j) for i, j in enumerate(result)]  # 准备排序
            no_sim_L = sorted(no_sim_L, key=lambda t: t[1], reverse=True)[:topK]  # 相似度越大越相似, 取topK

            s = sum([i for _, i in no_sim_L])  # 用于归一化
            if s == 0:  # 如果没有相似的
                prob = [1 / topK] * topK
            else:
                prob = [i / s for _, i in no_sim_L]
            paperID_probL_noL = (idL[index], prob, [i for i, _ in no_sim_L])  # 构建三元组结果
            概率矩阵.append(paperID_probL_noL)
        sys.stdout.write('\r')
        return 概率矩阵

    def compute_TtoT_prob(self, 进程数=2, 存储地址='', 含测试文本=True, topK=0.01):
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        idL = [i for i, _ in self._trainPaper_sentence_L]
        if 含测试文本:
            idL += [i for i, _ in self._testPaper_sentence_L]
            paperSen_csr_matrix = self._paperSen_csr_matrix
        else:
            paperSen_csr_matrix = self._trainPaperSen_csr_matrix
        if 0 < topK < 1:
            topK = round(topK * len(idL))
        elif topK <= 0:
            topK = len(idL)

        avgText = math.ceil(len(idL) / 进程数)
        参数l = [(paperSen_csr_matrix, idL, topK, avgText * i, avgText * (i+1)) for i in range(进程数)]
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

    def 计算分类数据集相似度矩阵(self, 进程数):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...')
        avgPaper = math.ceil(len(self._trainPaper_sentence_L)/进程数)
        queue = multiprocessing.Queue()
        已开启进程数 = 0
        for ID in range(进程数):
            multiprocessing.Process(target=self._cos_similarity, args=(self._trainPaperSen_csr_matrix[avgPaper*ID: avgPaper*(ID+1)], self._testPaperSen_csr_matrix, queue, ID)).start()
            已开启进程数 += 1
        id_result_D = {}
        for i in tqdm(range(已开启进程数), '进程运算中'):
            result, ID, _ = queue.get()
            id_result_D[ID] = result.tolist()
        train_test_sim_L = []  # 每一行表示一个训练集论文和所有测试集论文的相似度
        for i in range(len(id_result_D)):
            train_test_sim_L += id_result_D[i]
        test_train_dis_L = -np.array(train_test_sim_L).T  # 相似度越高距离越小

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

    TF_IDF_obj = TF_IDF(dataIRaddress=数据集地址)
    # TF_IDF_obj.compute_TtoT_prob(进程数=4,
    #                              存储地址='data/CR dblp/ac_allTtoT100-paperID_probL_noL_L.pkl',
    #                              含测试文本=True,
    #                              topK=100)  # 小于等于0表示取全部

    # 检索数据集评估
    test_train_D = TF_IDF_obj.计算测试集与训练集余弦相似度(进程数=3)

    IR评估_obj = IR评估(标签地址=数据集地址)
    IR评估_obj.评估(预测标签=test_train_D, topN=100, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_train_dis_L, trainLabel_L, testLabel_L = TF_IDF_obj.计算分类数据集相似度矩阵(进程数=8)
    # TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
