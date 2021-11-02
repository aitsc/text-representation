from tqdm import tqdm
import numpy as np
import multiprocessing
import time
from collections import Counter
from wmd import WMD
import importlib
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class WMDbased:
    def __init__(self, dataIRaddress, 词向量地址):
        print('数据集: %s' % dataIRaddress)
        self._trainPaper_sentence_L, self._testPaper_sentence_L = self._getPaperInfor(dataIRaddress)
        self._train_nbow, self._test_nbow, embeddings = self._get_nbow(self._trainPaper_sentence_L, self._testPaper_sentence_L, 词向量地址)
        self._wmdTrain = WMD(embeddings, self._train_nbow, vocabulary_min=2, vocabulary_max=1000)

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
    def _get_nbow(trainPaper_sentence_L, testPaper_sentence_L, 词向量地址):
        train_nbow = {}  # {文本编号:(文本编号, [词号,..], [词频,..]),..}  按词号顺序,词频是np.array
        test_nbow = {}  # {文本编号:(文本编号, [词号,..], [词频,..]),..}  按词号顺序,词频是np.array
        id_word_D = {}
        word_id_D = {}
        # 获取 nbow
        for 编号, 句子 in tqdm(trainPaper_sentence_L, '构建 train_nbow'):
            words = 句子.split()
            words_id = []
            for w in words:
                if w in word_id_D:
                    wID = word_id_D[w]
                else:
                    wID = len(id_word_D)
                    id_word_D[wID] = w
                    word_id_D[w] = wID
                words_id.append(wID)
            id_tf_L = sorted(Counter(words_id).items())
            train_nbow[编号] = (编号, [i for i, _ in id_tf_L], np.array([i for _, i in id_tf_L], dtype=np.float32))
        for 编号, 句子 in tqdm(testPaper_sentence_L, '构建 test_nbow'):
            words = 句子.split()
            words_id = []
            for w in words:
                if w in word_id_D:
                    wID = word_id_D[w]
                else:
                    wID = len(id_word_D)
                    id_word_D[wID] = w
                    word_id_D[w] = wID
                words_id.append(wID)
            id_tf_L = sorted(Counter(words_id).items())
            test_nbow[编号] = (编号, [i for i, _ in id_tf_L], np.array([i for _, i in id_tf_L], dtype=np.float32))
        # 获取 词向量
        embeddings = [None]*len(word_id_D)  # np.array , 与词序号对应
        n = 0
        with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8', errors='ignore') as r:
            for line in tqdm(r, '读取词向量'):
                line = line.strip().split(' ')
                if len(line) < 3:
                    continue
                word = line[0]
                if word in word_id_D:
                    embeddings[word_id_D[word]] = np.array([float(i) for i in line[1:]])
                    n += 1
        assert n == len(word_id_D), '文本包含词向量文件中没有的词!'
        return train_nbow, test_nbow, np.array(embeddings, dtype=np.float32)

    def statisticsWord(self):
        wordNum_max_min_avg = [0, 10**10, 0]
        词频_平均词种数_d = {}

        for _, (_, 词号l, 词频l) in tqdm(self._train_nbow.items(), '统计 self._train_nbow'):
            if len(词号l) > wordNum_max_min_avg[0]:
                wordNum_max_min_avg[0] = len(词号l)
            if len(词号l) < wordNum_max_min_avg[1]:
                wordNum_max_min_avg[1] = len(词号l)
            wordNum_max_min_avg[2] += len(词号l)
            for i, 词种数 in Counter(词频l).items():
                if i in 词频_平均词种数_d:
                    词频_平均词种数_d[i] += 词种数
                else:
                    词频_平均词种数_d[i] = 词种数
        for _, (_, 词号l, 词频l) in tqdm(self._test_nbow.items(), '统计 self._test_nbow'):
            if len(词号l) > wordNum_max_min_avg[0]:
                wordNum_max_min_avg[0] = len(词号l)
            if len(词号l) < wordNum_max_min_avg[1]:
                wordNum_max_min_avg[1] = len(词号l)
            wordNum_max_min_avg[2] += len(词号l)
            for i, 词种数 in Counter(词频l).items():
                if i in 词频_平均词种数_d:
                    词频_平均词种数_d[i] += 词种数
                else:
                    词频_平均词种数_d[i] = 词种数
        文本总数 = len(self._train_nbow) + len(self._test_nbow)
        wordNum_max_min_avg[2] /= (文本总数)
        for i in 词频_平均词种数_d.keys():
            词频_平均词种数_d[i] /= 文本总数
        词频_平均词种数_l = sorted(词频_平均词种数_d.items(), key=lambda t: t[1])
        print('wordNum_max_min_avg:%s' % str(wordNum_max_min_avg))
        print('词频_平均词种数_l:%s' % str(词频_平均词种数_l))

    def 词过滤(self, 过滤低频词=0, 停用词地址=None):
        if 停用词地址:
            stopwords_S = set()
            with open(停用词地址, 'r', encoding='utf-8') as r:
                for line in r:
                    line = line.strip()
                    stopwords_S.add(line)
                    stopwords_S.add(line.lower())
        else:
            stopwords_S = set()
        # 过滤 self._train_nbow
        for i, 编号_词号_词频_t in tqdm(self._train_nbow.items(), '过滤 self._train_nbow'):
            词号l, 词频l = [], []
            for x, y in zip(编号_词号_词频_t[1], 编号_词号_词频_t[2]):
                if y > 过滤低频词 and x not in stopwords_S:
                    词号l.append(x)
                    词频l.append(y)
            if len(词号l) > 0:
                self._train_nbow[i] = (编号_词号_词频_t[0], 词号l, 词频l)
        # 过滤 self._test_nbow
        for i, 编号_词号_词频_t in tqdm(self._test_nbow.items(), '过滤 self._test_nbow'):
            词号l, 词频l = [], []
            for x, y in zip(编号_词号_词频_t[1], 编号_词号_词频_t[2]):
                if y > 过滤低频词 and x not in stopwords_S:
                    词号l.append(x)
                    词频l.append(y)
            if len(词号l) > 0:
                self._test_nbow[i] = (编号_词号_词频_t[0], 词号l, 词频l)

    @staticmethod
    def _计算相似度_单进程(词号, 权重, wmdModel, queue: multiprocessing.Queue, 编号, topK):
        编号_距离l = wmdModel.nearest_neighbors((tuple(词号), list(权重)), k=topK, early_stop=1, max_time=36000, skipped_stop=1)
        queue.put((编号_距离l, 编号))

    def 计算测试集与训练集距离(self, 进程数=2, topK=20):
        test_train8sim_D = {}  # {测试集论文编号:[(训练集论文编号,距离),..],..}
        queue = multiprocessing.Queue()
        已开启进程数 = 0
        计数 = 0
        if not topK:
            topK = len(self._train_nbow)

        for 编号, 信息t in tqdm(self._test_nbow.items(), 'WMD距离计算'):
            multiprocessing.Process(target=self._计算相似度_单进程, args=(信息t[1], 信息t[2], self._wmdTrain, queue, 编号, topK)).start()
            已开启进程数 += 1
            计数 += 1
            while 已开启进程数 > 0 and (进程数 <= 已开启进程数 or 计数 == len(self._test_nbow)):
                训练集编号_距离l, 测试集编号 = queue.get()
                test_train8sim_D[测试集编号] = 训练集编号_距离l
                已开启进程数 -= 1

        # 排序
        for k in tqdm(test_train8sim_D.keys(), '排序'):
            test_train8sim_D[k] = sorted(test_train8sim_D[k], key=lambda t: t[1])  # 距离越小越相似
        self._test_train8sim_D = test_train8sim_D  # {测试集论文编号:[(训练集论文编号,距离),..],..}
        test_train_D = {k: [i[0] for i in v] for k, v in test_train8sim_D.items()}
        return test_train_D  # {测试集论文编号:[训练集论文编号,..]}

    def 计算分类数据集相似度矩阵(self, 进程数):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        test_train8sim_D = {}  # {测试集论文编号:[(训练集论文编号,距离),..],..}
        queue = multiprocessing.Queue()
        已开启进程数 = 0
        计数 = 0

        for 编号, 信息t in tqdm(self._test_nbow.items(), 'WMD距离计算'):
            multiprocessing.Process(target=self._计算相似度_单进程, args=(信息t[1], 信息t[2], self._wmdTrain, queue, 编号, len(self._train_nbow))).start()
            已开启进程数 += 1
            计数 += 1
            while 已开启进程数 > 0 and (进程数 <= 已开启进程数 or 计数 == len(self._test_nbow)):
                训练集编号_距离l, 测试集编号 = queue.get()
                test_train8sim_D[测试集编号] = 训练集编号_距离l
                已开启进程数 -= 1
        test_train_dis_L = [[None]*len(self._trainPaper_sentence_L) for _ in range(len(self._testPaper_sentence_L))]

        # 获取标签
        trainLabel_L = []
        testLabel_L = []
        trainID_no_D = {}
        testID_no_D = {}
        for textID, _ in self._trainPaper_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            trainLabel_L.append(label)
            trainID_no_D[textID] = len(trainID_no_D)
        for textID, _ in self._testPaper_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            testLabel_L.append(label)
            testID_no_D[textID] = len(testID_no_D)
        # 获取距离矩阵
        for testID, trainID_dis_L in tqdm(test_train8sim_D.items(), '获取距离矩阵'):
            dis_L = test_train_dis_L[testID_no_D[testID]]
            for trainID, dis in trainID_dis_L:
                dis_L[trainID_no_D[trainID]] = dis
        return test_train_dis_L, trainLabel_L, testLabel_L


if __name__ == '__main__':
    数据集地址 = 'data/TC IMDB/dataset.text'
    startTime = time.time()

    WMDbased_obj = WMDbased(dataIRaddress=数据集地址,
                            词向量地址='data/TC IMDB/ak_Corpus_vectors.text',
                            # 词向量地址='data/TC IMDB/ak_glove-Corpus_vectors.text',
                            )
    WMDbased_obj.词过滤(过滤低频词=1,  # 只有词频大于这个数的会保留
                     # 停用词地址='stopwords.text',
                     )
    WMDbased_obj.statisticsWord()
    # test_train_D = WMDbased_obj.计算测试集与训练集距离(进程数=31, topK=20)
    #
    # IR评估_obj = IR评估(标签地址=数据集地址)
    # IR评估_obj.评估(预测标签=test_train_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    test_train_dis_L, trainLabel_L, testLabel_L = WMDbased_obj.计算分类数据集相似度矩阵(进程数=60)
    TC评估.距离矩阵评估(test_train_dis_L, trainLabel_L, testLabel_L,
                n_neighbors=[3, 6, 9, 12, 15, 18],
                n_jobs=8, knn使用距离加权=True, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
