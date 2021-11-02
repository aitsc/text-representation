import torch
from pytorch_transformers import XLMTokenizer, XLMModel
import numpy as np
import math
import time
from tqdm import tqdm
from multiprocessing import Pool
import math
import pickle
import sys
import importlib
import logging
logging.basicConfig(level=logging.ERROR)  # 防止输出初始化句子过长的警告
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class XLM:
    def __init__(self, 数据集地址, bitchSize, GPU=True, senMaxLen=200, mask=True, pretrained_model='xlm-mlm-en-2048'):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(数据集地址)
        # 获取句向量
        self._trainSenVec, self._testSenVec = self._getAvgWordEmbedding(self._trainPaper_sentence_L, self._testPaper_sentence_L, GPU, bitchSize, senMaxLen, mask, pretrained_model)

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

    @staticmethod
    def _getAvgWordEmbedding(trainPaper_sentence_L, testPaper_sentence_L, GPU, bitchSize, senMaxLen, mask, pretrained_model):
        tokenizer = XLMTokenizer.from_pretrained(pretrained_model)
        model = XLMModel.from_pretrained(pretrained_model)
        model.eval()
        if GPU:
            model.to('cuda')

        sentences_L = []
        sentences_mask_L = []
        for _, sen in tqdm(trainPaper_sentence_L + testPaper_sentence_L, '初始化句子'):
            indexed_tokens = tokenizer.encode(sen)[:senMaxLen]  # 获得编号
            m = senMaxLen - len(indexed_tokens)  # 掩码长度
            indexed_tokens += [0] * m  # 补充掩码
            sentences_L.append(indexed_tokens)
            sentences_mask_L.append([1.] * (senMaxLen - m) + [0.] * m)

        bsNum = math.ceil(len(sentences_L) / bitchSize)
        senVectors_L = []
        with torch.no_grad():
            for i in tqdm(range(bsNum), '计算XLM平均词向量'):
                tokens_tensor = torch.tensor(sentences_L[i * bitchSize: (i + 1) * bitchSize])
                mask_tensor = torch.tensor(sentences_mask_L[i * bitchSize: (i + 1) * bitchSize])
                if GPU:
                    tokens_tensor = tokens_tensor.to('cuda')
                    mask_tensor = mask_tensor.to('cuda')
                if mask:  # 使用掩码将多余词屏蔽
                    hidden_states = model(tokens_tensor, attention_mask=mask_tensor)[0]
                    句向量 = (hidden_states * torch.unsqueeze(mask_tensor, 2)).sum(1) / torch.unsqueeze(mask_tensor.sum(1), 1)
                    句向量 = np.array(句向量.cpu())
                else:
                    hidden_states = model(tokens_tensor)[0]
                    句向量 = np.array((hidden_states.sum(1) / senMaxLen).cpu())
                senVectors_L.append(句向量)
        senVectors_L = np.vstack(senVectors_L)
        trainSenVec = senVectors_L[:len(trainPaper_sentence_L)]
        testSenVec = senVectors_L[len(trainPaper_sentence_L):]
        return trainSenVec, testSenVec  # np.array

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

    @staticmethod
    def _compute_TtoT_prob_singleProcess(args):
        textSenVec_N, idL, topK, indexL, indexR = args

        输出千分比 = 1
        概率矩阵 = []  # [(paperID,[prob,..],[no,..]),..], no表示在idL中的序号
        for i, index in enumerate(range(indexL, min(indexR, len(idL)))):
            permillage = int(i / (indexR - indexL) * 1000)
            if permillage > 输出千分比:
                输出千分比 = permillage
                sys.stdout.write('\r')
                print('%.1f%%---' % (permillage / 10,), end='')
                sys.stdout.flush()

            result = np.dot(textSenVec_N[index], textSenVec_N.T) / (np.linalg.norm(textSenVec_N[index]) * np.linalg.norm(textSenVec_N, axis=1))
            result = result.tolist()  # 一个相似度向量
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
            textSenVec_N = np.vstack([self._trainSenVec, self._testSenVec])
        else:
            textSenVec_N = self._trainSenVec
        if 0 < topK < 1:
            topK = round(topK * len(idL))
        elif topK <= 0:
            topK = len(idL)

        avgText = math.ceil(len(idL) / 进程数)
        参数l = [(textSenVec_N, idL, topK, avgText * i, avgText * (i+1)) for i in range(进程数)]
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

    XLM_obj = XLM(数据集地址=数据集地址, bitchSize=100, GPU=True, senMaxLen=200, mask=True, pretrained_model='xlm-mlm-en-2048')
    # XLM_obj.compute_TtoT_prob(进程数=8,
    #                           存储地址='data/IR arxiv/as_allTtoT100-paperID_probL_noL_L.pkl',
    #                           含测试文本=True,
    #                           topK=100)  # 小于等于0表示取全部

    # 检索数据集评估
    # test_train_D = XLM_obj.计算测试集与训练集余弦相似度()
    # IR评估_obj = IR评估(标签地址=数据集地址)
    # IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    test_train_dis_L, trainLabel_L, testLabel_L = XLM_obj.计算分类数据集相似度矩阵()
    TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
                n_neighbors=[3, 6, 9, 12, 15, 18],
                n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
