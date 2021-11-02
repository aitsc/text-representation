import os
import sys
from tqdm import tqdm


class IMDB:
    def __init__(self, trainPath, testPath):
        self._testId_class_D, self._testId_t1_t2 = self._getTest(testPath)
        self._trainId_t1_t2 = self._getTrain(trainPath)

    @staticmethod
    def _切分(text):
        text = text.split()
        t1 = ' '.join(text[:int(len(text)/2)])
        t2 = ' '.join(text[int(len(text)/2):])
        return t1, t2

    @staticmethod
    def _getTest(path):
        testId_class_D = {}
        testId_t1_t2 = []

        # 提取负例
        pathN = path + '/neg/'
        fileName_suffix = [os.path.splitext(i) for i in os.listdir(pathN)]
        fileName_suffix = [(i, j) for i, j in fileName_suffix if j.lower() == '.txt']
        for name, suffix in tqdm(fileName_suffix, '_getTest-提取负例'):
            with open(pathN+name+suffix, 'r', encoding='utf-8') as r:
                text = ' '.join([i.strip() for i in r.readlines()])
            testId = 'test__n__' + name
            testId_class_D[testId] = {'n', }
            testId_t1_t2.append([testId, *IMDB._切分(text)])

        # 提取正例
        pathP = path + '/pos/'
        fileName_suffix = [os.path.splitext(i) for i in os.listdir(pathP)]
        fileName_suffix = [(i, j) for i, j in fileName_suffix if j.lower() == '.txt']
        for name, suffix in tqdm(fileName_suffix, '_getTest-提取正例'):
            with open(pathP+name+suffix, 'r', encoding='utf-8') as r:
                text = ' '.join([i.strip() for i in r.readlines()])
            testId = 'test__p__' + name
            testId_class_D[testId] = {'p', }
            testId_t1_t2.append([testId, *IMDB._切分(text)])
        return testId_class_D, testId_t1_t2

    @staticmethod
    def _getTrain(path):
        trainId_t1_t2 = []

        # 提取负例
        pathN = path + '/neg/'
        fileName_suffix = [os.path.splitext(i) for i in os.listdir(pathN)]
        fileName_suffix = [(i, j) for i, j in fileName_suffix if j.lower() == '.txt']
        for name, suffix in tqdm(fileName_suffix, '_getTrain-提取负例'):
            with open(pathN+name+suffix, 'r', encoding='utf-8') as r:
                text = ' '.join([i.strip() for i in r.readlines()])
            testId = 'train__n__' + name
            trainId_t1_t2.append([testId, *IMDB._切分(text)])

        # 提取正例
        pathP = path + '/pos/'
        fileName_suffix = [os.path.splitext(i) for i in os.listdir(pathP)]
        fileName_suffix = [(i, j) for i, j in fileName_suffix if j.lower() == '.txt']
        for name, suffix in tqdm(fileName_suffix, '_getTrain-提取正例'):
            with open(pathP+name+suffix, 'r', encoding='utf-8') as r:
                text = ' '.join([i.strip() for i in r.readlines()])
            testId = 'train__p__' + name
            trainId_t1_t2.append([testId, *IMDB._切分(text)])
        return trainId_t1_t2

    def statistics(self):
        testWord_max_min_avg = [0, 10**8, 0]
        trainWord_max_min_avg = [0, 10**8, 0]
        for _, t1, t2 in self._testId_t1_t2:
            length = len(t1.split()) + len(t2.split())
            if testWord_max_min_avg[0] < length:
                testWord_max_min_avg[0] = length
            if testWord_max_min_avg[1] > length:
                testWord_max_min_avg[1] = length
            testWord_max_min_avg[2] += length
        testWord_max_min_avg[2] /= len(self._testId_t1_t2)
        for _, t1, t2 in self._trainId_t1_t2:
            length = len(t1.split()) + len(t2.split())
            if trainWord_max_min_avg[0] < length:
                trainWord_max_min_avg[0] = length
            if trainWord_max_min_avg[1] > length:
                trainWord_max_min_avg[1] = length
            trainWord_max_min_avg[2] += length
        trainWord_max_min_avg[2] /= len(self._trainId_t1_t2)
        print('testWord_max_min_avg:%s, trainWord_max_min_avg:%s, 训练集数量:%d, 测试集数量:%d' %
              (str(testWord_max_min_avg), str(trainWord_max_min_avg), len(self._trainId_t1_t2), len(self._testId_t1_t2)))

    def saveDataset(self, address, segTitleAbstract='\t', lower=True):
        assert self._trainId_t1_t2, '没有训练集信息可以存储!'
        assert self._testId_t1_t2, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        testPatent_label_D = self._testId_class_D  # {测试集专利:训练集专利答案set,..}
        with open(address, 'w', encoding='utf-8') as w:
            w.write(str(testPatent_label_D)+'\n')
            for patentID, t1, t2 in tqdm(self._trainId_t1_t2, mName + '-写入训练集专利文本'):
                text = t1 + segTitleAbstract + t2
                if lower:
                    text = text.lower()
                w.write(patentID + '\t' + text + '\n')
            for patentID, t1, t2 in tqdm(self._testId_t1_t2, mName + '-写入测试集专利文本'):
                text = t1 + segTitleAbstract + t2
                if lower:
                    text = text.lower()
                w.write(patentID + '\t' + text + '\n')


if __name__ == '__main__':
    IMDB_obj = IMDB(trainPath=r'F:\data\aclImdb\train',
                    testPath=r'F:\data\aclImdb\test')
    IMDB_obj.statistics()
    IMDB_obj.saveDataset(address=r'data/TC IMDB/dataset.text', lower=True)
