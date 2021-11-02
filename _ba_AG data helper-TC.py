import sys
from tqdm import tqdm


class AGnews:
    def __init__(self, trainPath, testPath):
        self._testId_class_D, self._testId_t1_t2 = self._getTest(testPath)
        self._trainId_t1_t2 = self._getTrain(trainPath)

    @staticmethod
    def _getTest(path):
        testId_class_D = {}
        testId_t1_t2 = []

        with open(path, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '_getTest'):
                c, t1, t2 = line.strip()[1:-1].split('","')
                testId = 'test__%s__' % c + str(i)
                testId_class_D[testId] = {c, }
                testId_t1_t2.append([testId, t1, t2])
        return testId_class_D, testId_t1_t2

    @staticmethod
    def _getTrain(path):
        trainId_t1_t2 = []

        with open(path, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '_getTrain'):
                c, t1, t2 = line.strip()[1:-1].split('","')
                testId = 'train__%s__' % c + str(i)
                trainId_t1_t2.append([testId, t1, t2])
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
            for patentID, t1, t2 in tqdm(self._trainId_t1_t2, mName + '-写入训练集文本'):
                text = t1 + segTitleAbstract + t2
                if lower:
                    text = text.lower()
                w.write(patentID + '\t' + text + '\n')
            for patentID, t1, t2 in tqdm(self._testId_t1_t2, mName + '-写入测试集文本'):
                text = t1 + segTitleAbstract + t2
                if lower:
                    text = text.lower()
                w.write(patentID + '\t' + text + '\n')


if __name__ == '__main__':
    AGnews_obj = AGnews(trainPath=r"F:\data\ag_news_csv\train.csv",
                        testPath=r"F:\data\ag_news_csv\test.csv")
    # AGnews_obj.statistics()
    AGnews_obj.saveDataset(address=r'data/TC AGnews/dataset.text', lower=True)
