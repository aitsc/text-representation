import numpy as np
import time
from tqdm import tqdm
import h5py
import random
import re
import pickle
import importlib
import os
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class SentEmbedding:
    class Model_avgWord:
        def __init__(self, sent_dim, wordEmPath='', bitchSize=1000, textCleaning=True, name=''):
            '''
            :param sent_dim:
            :param wordEmPath: 词向量的第一行不是向量
            :param bitchSize:
            :param textCleaning:
            :param name:
            '''
            self.wordEmPath = wordEmPath  # 为空则是随机词向量
            self.bitchSize = bitchSize
            self.textCleaning = textCleaning
            self.sent_dim = sent_dim
            if wordEmPath:
                self.name = 'avgWord' + name
            else:
                self.name = 'avgRandomWord'
        ID = 'Model_avgWord'

    class Model_readSent:
        def __init__(self, sent_dim, sentEmPath='', bitchSize=1000, textCleaning=True, name=''):
            '''
            :param sent_dim:
            :param sentEmPath: 文件格式: {编号:[[前部句向量,..],[后部句向量,..]],..}
            :param bitchSize:
            :param textCleaning:
            :param name:
            '''
            self.sentEmPath = sentEmPath  # 为空则是随机句向量
            self.bitchSize = bitchSize
            self.textCleaning = textCleaning
            self.sent_dim = sent_dim
            if sentEmPath:
                self.name = 'readSent' + name
            else:
                self.name = 'randomSent'
        ID = 'Model_readSent'

    def __init__(self, datasetPath, Model_para, senFrontMaxLen=20, senBackMaxLen=20, senMatrixPath=None, model_F=None):
        print('数据: ' + datasetPath)
        if Model_para.ID == self.Model_avgWord.ID:
            self._Model_para = Model_para
            self._getSenVectors_F = self._model_avgWord
        elif Model_para.ID == self.Model_readSent.ID:
            self._Model_para = Model_para
            self._getSenVectors_F = self._model_readSent
        elif model_F:
            self._Model_para = Model_para
            self._getSenVectors_F = model_F
        else:
            raise Exception("缺乏有效的模型!", datasetPath, model_F)
        try:
            textCleaning = self._Model_para.textCleaning
        except:
            textCleaning = True
        self._Model_para.senFrontMaxLen = senFrontMaxLen
        self._Model_para.senBackMaxLen = senBackMaxLen

        self._candTextF_sentence_L, self._candTextB_sentence_L, self._testTextF_sentence_L, self._testTextB_sentence_L = self._getTextInfor(datasetPath, textCleaning)
        self._senID_mat_len_seg_mid_h5, self._senID_no_D = self._readSenVectors(senMatrixPath, self._Model_para.sent_dim, senFrontMaxLen, senBackMaxLen)

        self._candSenVec, self._testSenVec = self._getSenVectors_F(self._Model_para, self._saveSentEmbedding, self._candTextF_sentence_L, self._candTextB_sentence_L, self._testTextF_sentence_L, self._testTextB_sentence_L)

    def _getTextInfor(self, dataIRaddress, textCleaning):
        candTextF_sentence_L = []  # [(候选集id,[文本前部]),..], 文本前部和后部是没有清洗的句子
        candTextB_sentence_L = []  # [(候选集id,[文本后部]),..]
        testTextF_sentence_L = []  # [(测试集id,[文本前部]),..]
        testTextB_sentence_L = []  # [(测试集id,[文本后部]),..]
        test_candidateS_D = {}  # {测试集id:候选集id set,..}

        segDocToSent = lambda t: self._segDocToSent(t)
        if textCleaning:
            s = lambda t: [句子清洗(j) for j in self._segDocToSent(t)]
            segDocToSent = lambda t: [k for k in s(t) if k]

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取文本信息'):
                if i == 0:
                    test_candidateS_D = eval(line.strip())
                    continue
                line = line.strip().split('\t')
                textID = line[0]
                if textID in test_candidateS_D:
                    testTextF_sentence_L.append((textID, segDocToSent(line[1])))
                    testTextB_sentence_L.append((textID, segDocToSent(line[2])))
                else:
                    candTextF_sentence_L.append((textID, segDocToSent(line[1])))
                    candTextB_sentence_L.append((textID, segDocToSent(line[2])))
        return candTextF_sentence_L, candTextB_sentence_L, testTextF_sentence_L, testTextB_sentence_L

    def saveTextInfor(self, path, protocol=2):
        '''
        为了 _bg_skip thoughts.py
        :param path:
        :param protocol:
        :return:
        '''
        assert self._candTextF_sentence_L, '没有 self._candTextF_sentence_L'
        print('输出 id_ft8bt_D')
        id_ft8bt_D = {}  # {编号:[[前部句,..],[后部句,..]],..}
        for (id_f, sen_f), (id_b, sen_b) in zip(self._candTextF_sentence_L, self._candTextB_sentence_L):
            assert id_f == id_b, '前部 后部 编号不对应!'
            id_ft8bt_D[id_f] = [sen_f, sen_b]
        for (id_f, sen_f), (id_b, sen_b) in zip(self._testTextF_sentence_L, self._testTextB_sentence_L):
            assert id_f == id_b, '前部 后部 编号不对应!'
            id_ft8bt_D[id_f] = [sen_f, sen_b]

        info = pickle.dumps(id_ft8bt_D, protocol=protocol)
        with open(path, 'wb') as w:
            w.write(info)

    @staticmethod
    def _readSenVectors(senMatrixPath, word_dim, senFrontMaxLen, senBackMaxLen):
        if not senMatrixPath:
            return None, None
        senID_mat_len_seg_mid_h5 = h5py.File(senMatrixPath.encode('utf-8'), 'w')
        senMaxLen = senFrontMaxLen + senBackMaxLen
        if 'senID' in senID_mat_len_seg_mid_h5:
            senID_no_D = {j: i for i, j in enumerate(senID_mat_len_seg_mid_h5['senID']) if j}
            assert senID_mat_len_seg_mid_h5['matrix'].shape[1] == senMaxLen, '句矩阵与模型句子长度不一致!'
            assert senID_mat_len_seg_mid_h5['matrix'].shape[2] == word_dim, '句矩阵与模型词维度不一致!'
            assert senID_mat_len_seg_mid_h5['senFrontMaxLen'][0] == senFrontMaxLen, '句矩阵与模型senFrontMaxLen不一致!'
            assert senID_mat_len_seg_mid_h5['senBackMaxLen'][0] == senBackMaxLen, '句矩阵与模型senBackMaxLen不一致!'
        else:
            senID_no_D = {}
            senID_mat_len_seg_mid_h5.create_dataset("senID", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
            senID_mat_len_seg_mid_h5.create_dataset("matrix", (0, senMaxLen, word_dim), maxshape=(None, senMaxLen, word_dim), dtype=np.float32, chunks=(1, senMaxLen, word_dim))
            senID_mat_len_seg_mid_h5.create_dataset("length", (0,), maxshape=(None,), dtype=np.int32, chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("segPos", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('float32')), chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("mid", (0,), maxshape=(None,), dtype=np.int32, chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("senFrontMaxLen", data=[senFrontMaxLen])
            senID_mat_len_seg_mid_h5.create_dataset("senBackMaxLen", data=[senBackMaxLen])
        return senID_mat_len_seg_mid_h5, senID_no_D

    def _saveSentEmbedding(self, id_L, senMatrix, sentences_mask_L, segPos_L, mid_L):
        senVec, testSenVec = [], []
        tokenNum_max_min_avg = [0, 10**10, 0]
        sentences_mask_L = np.sum(sentences_mask_L, axis=1)

        if self._senID_mat_len_seg_mid_h5:
            senID = self._senID_mat_len_seg_mid_h5['senID']
            matrix = self._senID_mat_len_seg_mid_h5['matrix']
            length = self._senID_mat_len_seg_mid_h5['length']
            segPos = self._senID_mat_len_seg_mid_h5['segPos']
            mid = self._senID_mat_len_seg_mid_h5['mid']
            # 增加大小
            senID.resize([senID.shape[0] + len(id_L)])
            matrix.resize([matrix.shape[0] + len(id_L), matrix.shape[1], matrix.shape[2]])
            length.resize([length.shape[0] + len(id_L)])
            segPos.resize([segPos.shape[0] + len(id_L)])
            mid.resize([mid.shape[0] + len(id_L)])

            senID[-len(id_L):] = id_L
            matrix[-len(id_L):] = senMatrix
            length[-len(id_L):] = sentences_mask_L
            segPos[-len(id_L):] = segPos_L
            mid[-len(id_L):] = mid_L
            self._senID_mat_len_seg_mid_h5.flush()
            for j in id_L:  # 如果有重复索引会导致一些冗余
                self._senID_no_D[j] = len(self._senID_no_D)
        for j in range(len(id_L)):  # 获得句向量
            l = sentences_mask_L[j]
            senVec.append(np.sum(senMatrix[j], axis=0) / l)
            if l > tokenNum_max_min_avg[0]:
                tokenNum_max_min_avg[0] = l
            if l < tokenNum_max_min_avg[1]:
                tokenNum_max_min_avg[1] = l
            tokenNum_max_min_avg[2] += l
        tokenNum_max_min_avg[2] /= len(id_L)

        return np.array(senVec), tokenNum_max_min_avg

    def _segDocToSent(self, text):
        text = re.split('[.!?] +', text)
        text_ = []
        for i in text:
            i = i.strip()
            if i:
                text_.append(i)
        return text_

    @staticmethod
    def _model_avgWord(Model_para, saveSentEmbedding, candTextF_sentence_L, candTextB_sentence_L, testTextF_sentence_L, testTextB_sentence_L):
        candSenVec, testSenVec = [], []
        tokenNum_max_min = [0, 10**10]
        senF_L = candTextF_sentence_L + testTextF_sentence_L  # [(候选集id,[文本前部句子,..]),..]
        senB_L = candTextB_sentence_L + testTextB_sentence_L
        senF_L = [(i, j[:Model_para.senFrontMaxLen]) for i, j in senF_L]
        senB_L = [(i, j[:Model_para.senBackMaxLen]) for i, j in senB_L]
        senMaxLen = Model_para.senFrontMaxLen + Model_para.senBackMaxLen

        all_words = set()  # 获得所有词集合
        for _, i in senF_L + senB_L:
            for j in i:
                all_words |= set(j.split())
        # 读取所有词向量
        词_向量d = {}
        维度 = -1
        if Model_para.wordEmPath:
            with open(Model_para.wordEmPath.encode('utf-8'), 'r', encoding='utf-8', errors='ignore') as r:
                for line in tqdm(r, '读取词向量'):
                    line = line.strip().split(' ')
                    if len(line) < 3:
                        continue
                    word = line[0]
                    if word in all_words:
                        词_向量d[word] = np.array([float(i) for i in line[1:]])
                        if 维度 <= 0:
                            维度 = len(词_向量d[word])
            assert 维度 == Model_para.sent_dim, '词向量和句向量维度不一致!'
            print('有%d种词不在词向量文件中' % (len(all_words)-len(词_向量d)))
        else:  # 随机词向量
            维度 = Model_para.sent_dim
            for w in tqdm(all_words, '构建随机词向量'):
                词_向量d[w] = [random.random() for _ in range(维度)]

        for i in tqdm(range(0, len(senF_L), Model_para.bitchSize), '计算句向量'):
            id_L = []
            senFv_L, senBv_L = [], []  # [[文本部句嵌入_L,..],..]
            for (id_f, sen_f), (id_b, sen_b) in zip(senF_L[i: i+Model_para.bitchSize], senB_L[i: i+Model_para.bitchSize]):
                assert id_f == id_b, '前部 后部 编号不对应!'
                id_L.append(id_f)
                senFv_L.append([])
                for sen in sen_f:
                    w_num = 0
                    senFv_L[-1].append(np.array([0.]*维度))
                    for w in sen.split():
                        if w in 词_向量d:
                            w_num += 1
                            senFv_L[-1][-1] += 词_向量d[w]
                    senFv_L[-1][-1] /= w_num
                senBv_L.append([])
                for sen in sen_b:
                    w_num = 0
                    senBv_L[-1].append(np.array([0.]*维度))
                    for w in sen.split():
                        if w in 词_向量d:
                            w_num += 1
                            senBv_L[-1][-1] += 词_向量d[w]
                    senBv_L[-1][-1] /= w_num
            senMatrix, sentences_mask_L, segPos_L, mid_L = [], [], [], []
            for f, b in zip(senFv_L, senBv_L):
                l = len(f) + len(b)
                senMatrix.append(f+b+[[0.]*维度 for _ in range(senMaxLen-l)])
                sentences_mask_L.append([1.]*l+[0.]*(senMaxLen-l))
                segPos_L.append([i for i in range(l + 1)])
                mid_L.append(len(f))
            senMatrix, sentences_mask_L, segPos_L, mid_L = np.array(senMatrix), np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)

            candSenVec_, tokenNum_max_min_avg_ = saveSentEmbedding(id_L, senMatrix, sentences_mask_L, segPos_L, mid_L)
            candSenVec.append(candSenVec_)
            if tokenNum_max_min[0] < tokenNum_max_min_avg_[0]:
                tokenNum_max_min[0] = tokenNum_max_min_avg_[0]
            if tokenNum_max_min[1] > tokenNum_max_min_avg_[1]:
                tokenNum_max_min[1] = tokenNum_max_min_avg_[1]

        print('tokenNum_max_min: %s' % str(tokenNum_max_min))
        candSenVec = np.vstack(candSenVec)
        testSenVec = candSenVec[len(candTextF_sentence_L):]  # 和下面顺序不能反
        candSenVec = candSenVec[:len(candTextF_sentence_L)]
        return candSenVec, testSenVec

    @staticmethod
    def _model_readSent(Model_para, saveSentEmbedding, candTextF_sentence_L, candTextB_sentence_L, testTextF_sentence_L, testTextB_sentence_L):
        candSenVec, testSenVec = [], []
        tokenNum_max_min = [0, 10**10]
        senF_L = candTextF_sentence_L + testTextF_sentence_L  # [(候选集id,[文本前部句子,..]),..]
        senB_L = candTextB_sentence_L + testTextB_sentence_L
        senF_L = [(i, j[:Model_para.senFrontMaxLen]) for i, j in senF_L]
        senB_L = [(i, j[:Model_para.senBackMaxLen]) for i, j in senB_L]
        senMaxLen = Model_para.senFrontMaxLen + Model_para.senBackMaxLen

        # 读取所有句向量
        if Model_para.sentEmPath:
            print('读取所有句向量...')
            try:
                with open(Model_para.sentEmPath.encode('utf-8'), 'rb') as r:
                    id_ftV8btV_D = pickle.load(r)  # {编号:[[前部句向量,..],[后部句向量,..]],..}
            except:
                with open(Model_para.sentEmPath.encode('utf-8'), 'rb') as r:
                    id_ftV8btV_D = pickle.load(r, encoding='iso-8859-1')  # 兼容 python2
            for i in id_ftV8btV_D.values():
                维度 = len(i[0][0])
                break
            assert 维度 == Model_para.sent_dim, '句向量和文档向量维度不一致!'
            assert len(senF_L) == len(id_ftV8btV_D), '样本数量不一致!'
        else:  # 随机句向量
            句_向量d = {}
            all_sents = set()  # 获得所有句集合
            for _, i in senF_L + senB_L:
                all_sents |= set(i)
            维度 = Model_para.sent_dim
            for s in tqdm(all_sents, '构建随机句向量'):
                句_向量d[s] = [random.random() for _ in range(维度)]

        for i in tqdm(range(0, len(senF_L), Model_para.bitchSize), '获得句向量'):
            id_L = []
            senFv_L, senBv_L = [], []  # [[文本部句嵌入_L,..],..]
            for (id_f, sen_f), (id_b, sen_b) in zip(senF_L[i: i+Model_para.bitchSize], senB_L[i: i+Model_para.bitchSize]):
                assert id_f == id_b, '前部 后部 编号不对应!'
                if len(sen_f) + len(sen_b) > senMaxLen:
                    print(len(sen_f), len(sen_b))
                    print(sen_f)
                    print(sen_b)
                id_L.append(id_f)
                if Model_para.sentEmPath:
                    senFv_L.append(id_ftV8btV_D[id_f][0][:Model_para.senFrontMaxLen])
                    senBv_L.append(id_ftV8btV_D[id_f][1][:Model_para.senBackMaxLen])
                else:
                    senFv_L.append([])
                    for sen in sen_f:
                        senFv_L[-1].append(句_向量d[sen])
                    senBv_L.append([])
                    for sen in sen_b:
                        senBv_L[-1].append(句_向量d[sen])
            senMatrix, sentences_mask_L, segPos_L, mid_L = [], [], [], []
            for f, b in zip(senFv_L, senBv_L):
                l = len(f) + len(b)
                senMatrix.append(f+b+[[0.]*维度 for _ in range(senMaxLen-l)])
                sentences_mask_L.append([1.]*l+[0.]*(senMaxLen-l))
                segPos_L.append([i for i in range(l + 1)])
                mid_L.append(len(f))
            senMatrix = np.array(senMatrix)
            sentences_mask_L = np.array(sentences_mask_L)
            segPos_L = np.array(segPos_L)
            mid_L = np.array(mid_L)

            candSenVec_, tokenNum_max_min_avg_ = saveSentEmbedding(id_L, senMatrix, sentences_mask_L, segPos_L, mid_L)
            candSenVec.append(candSenVec_)
            if tokenNum_max_min[0] < tokenNum_max_min_avg_[0]:
                tokenNum_max_min[0] = tokenNum_max_min_avg_[0]
            if tokenNum_max_min[1] > tokenNum_max_min_avg_[1]:
                tokenNum_max_min[1] = tokenNum_max_min_avg_[1]

        print('tokenNum_max_min: %s' % str(tokenNum_max_min))
        candSenVec = np.vstack(candSenVec)
        testSenVec = candSenVec[len(candTextF_sentence_L):]  # 和下面顺序不能反
        candSenVec = candSenVec[:len(candTextF_sentence_L)]
        return candSenVec, testSenVec

    def computeIRcosSimMatrix(self):
        print('cos计算中...')
        # 每一行表示一个候选集文本和所有测试集文本的相似度
        cand_test_sim_L = (np.dot(self._testSenVec, self._candSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._candSenVec, axis=1), axis=0))).T

        # 还原编号
        test_cand8sim_D = {}  # {测试集文本编号:{候选集文本编号:相似度,..},..}
        for i in tqdm(range(len(cand_test_sim_L)), '还原编号, 候选集数'):
            for j, sim in enumerate(cand_test_sim_L[i]):
                testID = self._testTextF_sentence_L[j][0]
                candID = self._candTextF_sentence_L[i][0]
                if testID in test_cand8sim_D:
                    test_cand8sim_D[testID][candID] = sim
                else:
                    test_cand8sim_D[testID] = {candID: sim}
        # 排序
        for k in tqdm(test_cand8sim_D.keys(), '排序'):
            test_cand8sim_D[k] = sorted(test_cand8sim_D[k].items(), key=lambda t: t[1], reverse=True)
        self._test_cand8sim_D = test_cand8sim_D  # {测试集文本编号:[(候选集文本编号,相似度),..],..}
        test_cand_D = {k: [i[0] for i in v] for k, v in test_cand8sim_D.items()}
        return test_cand_D  # {测试集文本编号:[候选集文本编号,..]}

    def computeTCdisMatrix(self):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        print('dis计算中...')
        # 每一行表示一个候选集文本和所有测试集文本的距离(相似度的负数)
        test_cand_dis_L = -np.dot(self._testSenVec, self._candSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._candSenVec, axis=1), axis=0))

        # 获取标签
        candLabel_L = []
        testLabel_L = []
        for textID, _ in self._candTextF_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            candLabel_L.append(label)
        for textID, _ in self._testTextF_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            testLabel_L.append(label)
        return test_cand_dis_L, candLabel_L, testLabel_L

    # def __del__(self):
    #     if self._senID_mat_len_seg_mid_h5:
    #         self._senID_mat_len_seg_mid_h5.close()  # ImportError: sys.meta_path is None, Python is likely shutting down ?


if __name__ == '__main__':
    # ap = 'data/IR arxiv/'
    ap = 'data/CR USPTO/'

    datasetPath = ap + 'dataset.text'
    senMatrixFolder = r'F:\data\_large_tem_file\CTE/' + ap
    # senMatrixFolder = ''
    if not senMatrixFolder or not os.path.exists(senMatrixFolder):
        senMatrixFolder = ap

    # Model_para = SentEmbedding.Model_avgWord(sent_dim=100, wordEmPath='', bitchSize=1000, textCleaning=True)
    # Model_para = SentEmbedding.Model_readSent(sent_dim=100, sentEmPath='', bitchSize=1000, textCleaning=True)
    # Model_para = SentEmbedding.Model_avgWord(sent_dim=100, wordEmPath=ap+'ak_Corpus_vectors.text', bitchSize=1000, textCleaning=True, name='-w2v')
    Model_para = SentEmbedding.Model_readSent(sent_dim=4800, sentEmPath=ap+'bg_id_ftV8btV_D.pkl', bitchSize=100, textCleaning=True, name='-skip')

    senMatrixPath = senMatrixFolder + 'bf_' + Model_para.name + '_senID_mat_len_seg_mid.h5'
    startTime = time.time()
    SentEmbedding_obj = SentEmbedding(datasetPath=datasetPath,
                                      Model_para=Model_para,
                                      senFrontMaxLen=20,
                                      senBackMaxLen=20,
                                      senMatrixPath=senMatrixPath,
                                      model_F=None,
                                      )
    SentEmbedding_obj.saveTextInfor(path=ap+'bf_id_ft8bt_D.pkl', protocol=2)  # 用于 skip-thougths

    # 检索数据集评估
    test_cand_D = SentEmbedding_obj.computeIRcosSimMatrix()
    IR评估_obj = IR评估(标签地址=datasetPath)
    IR评估_obj.评估(预测标签=test_cand_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_cand_dis_L, candLabel_L, testLabel_L = SentEmbedding_obj.computeTCdisMatrix()
    # TC评估.距离矩阵评估(test_cand_dis_L, candLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
