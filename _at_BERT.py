import torch
from pytorch_transformers import BertTokenizer, BertModel
import numpy as np
import math
import time
from tqdm import tqdm
import h5py
import importlib
import logging
logging.basicConfig(level=logging.ERROR)  # 防止输出初始化句子过长的警告
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class BERT:
    def __init__(self, 数据集地址, bitchSize, GPU=True, senMaxLen=200, mask=True, pretrained_model='bert-large-uncased',
                 句矩阵文件地址=None):
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(数据集地址)
        # 获取句向量
        if not 句矩阵文件地址:  # 获取句向量
            self._trainSenVec, self._testSenVec = self._getAvgWordEmbedding(self._trainPaper_sentence_L, self._testPaper_sentence_L, GPU, bitchSize, senMaxLen, mask, pretrained_model)
        else:
            self._trainSenVec, self._testSenVec = self._预加载数据集和句矩阵(数据集地址, 句矩阵文件地址)

    @staticmethod
    def _预加载数据集和句矩阵(数据集地址, 句矩阵文件地址):
        print('读取句矩阵文件地址...')
        trainPaperF_sentence_L = []  # [(训练集论文前部,句子),..]
        trainPaperB_sentence_L = []  # [(训练集论文后部,句子),..]
        testPaperF_sentence_L = []  # [(测试集论文前部,句子),..]
        testPaperB_sentence_L = []  # [(测试集论文后部,句子),..]
        testPaper_IRpaper_D = {}  # {测试集论文:训练集论文set,..}

        with open(数据集地址, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取论文信息'):
                if i == 0:
                    testPaper_IRpaper_D = eval(line.strip())
                    continue
                line = line.split('\t')
                paperID = line[0]
                if paperID in testPaper_IRpaper_D:
                    testPaperF_sentence_L.append((paperID, 句子清洗(line[1])))
                    testPaperB_sentence_L.append((paperID, 句子清洗(line[2])))
                else:
                    trainPaperF_sentence_L.append((paperID, 句子清洗(line[1])))
                    trainPaperB_sentence_L.append((paperID, 句子清洗(line[2])))

        句子_矩阵_长度h5 = h5py.File(句矩阵文件地址.encode('utf-8'), 'a')
        句子_no_D = {j: i for i, j in enumerate(句子_矩阵_长度h5['sentences'])}
        matrixs_L = 句子_矩阵_长度h5['matrixs']
        lengths_L = 句子_矩阵_长度h5['lengths']
        trainSenVec = []
        for (_, f), (_, b) in tqdm(zip(trainPaperF_sentence_L, trainPaperB_sentence_L), '构建训练集句向量'):
            f, b = 句子_no_D[f], 句子_no_D[b]
            # sen = matrixs_L[f].sum(0) / lengths_L[f]  # 单独前部
            # sen = matrixs_L[b].sum(0) / lengths_L[b]  # 单独后部
            sen = (matrixs_L[f].sum(0) + matrixs_L[b].sum(0)) / (lengths_L[f] + lengths_L[b])
            trainSenVec.append(sen)
        testSenVec = []
        for (_, f), (_, b) in tqdm(zip(testPaperF_sentence_L, testPaperB_sentence_L), '构建测试集句向量'):
            f, b = 句子_no_D[f], 句子_no_D[b]
            # sen = matrixs_L[f].sum(0) / lengths_L[f]  # 单独前部
            # sen = matrixs_L[b].sum(0) / lengths_L[b]  # 单独后部
            sen = (matrixs_L[f].sum(0) + matrixs_L[b].sum(0)) / (lengths_L[f] + lengths_L[b])
            testSenVec.append(sen)
        return np.array(trainSenVec), np.array(testSenVec)  # np.array

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
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        model = BertModel.from_pretrained(pretrained_model)
        model.eval()
        if GPU:
            model.to('cuda')

        sentences_L = []
        sentences_mask_L = []
        SEP = tokenizer.convert_tokens_to_ids('[SEP]')
        for _, sen in tqdm(trainPaper_sentence_L + testPaper_sentence_L, '初始化句子'):
            indexed_tokens = tokenizer.encode('[CLS] ' + sen)[: senMaxLen - 1]  # 获得编号
            indexed_tokens.append(SEP)  # 加入 [SEP]
            m = senMaxLen - len(indexed_tokens)  # 掩码长度
            indexed_tokens += [0] * m  # 补充掩码
            sentences_L.append(indexed_tokens)
            sentences_mask_L.append([1.] * (senMaxLen - m) + [0.] * m)

        bsNum = math.ceil(len(sentences_L) / bitchSize)
        senVectors_L = []
        with torch.no_grad():
            for i in tqdm(range(bsNum), '计算BERT平均词向量'):
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

    BERT_obj = BERT(数据集地址=数据集地址, bitchSize=100, GPU=True, senMaxLen=400, mask=True,
                    # pretrained_model='bert-large-uncased-whole-word-masking',
                    pretrained_model='bert-large-uncased',
                    句矩阵文件地址=r"F:\data\_large_tem_file\CTE\data\CR dblp\av_bert-large-uncased_sen_mat_len.h5",
                    )

    # 检索数据集评估
    test_train_D = BERT_obj.计算测试集与训练集余弦相似度()
    IR评估_obj = IR评估(标签地址=数据集地址)
    IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_train_dis_L, trainLabel_L, testLabel_L = BERT_obj.计算分类数据集相似度矩阵()
    # TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
