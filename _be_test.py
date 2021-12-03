from tqdm import tqdm
import h5py
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 比较不同h5py:
    def __init__(self, oldH5, newH5, datasetPath):
        self._oldH5 = h5py.File(oldH5.encode('utf-8'), 'r')
        self._oldH5_senID_no_D = {j: i for i, j in enumerate(self._oldH5['sentences']) if j}
        self._newH5 = h5py.File(newH5.encode('utf-8'), 'r')
        self._newH5_senID_no_D = {j: i for i, j in enumerate(self._newH5['senID']) if j}
        self._candTextF_sentence_L, self._candTextB_sentence_L, self._testTextF_sentence_L, self._testTextB_sentence_L = self._getTextInfor(datasetPath)

    @staticmethod
    def _getTextInfor(dataIRaddress):
        candTextF_sentence_L = []  # [(候选集id,[文本前部]),..]
        candTextB_sentence_L = []  # [(候选集id,[文本后部]),..]
        testTextF_sentence_L = []  # [(测试集id,[文本前部]),..]
        testTextB_sentence_L = []  # [(测试集id,[文本后部]),..]
        test_candidateS_D = {}  # {测试集id:候选集id set,..}

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取文本信息'):
                if i == 0:
                    test_candidateS_D = eval(line.strip())
                    continue
                line = line.split('\t')
                textID = line[0]
                if textID in test_candidateS_D:
                    testTextF_sentence_L.append((textID, 句子清洗(line[1]).split()))
                    testTextB_sentence_L.append((textID, 句子清洗(line[2]).split()))
                else:
                    candTextF_sentence_L.append((textID, 句子清洗(line[1]).split()))
                    candTextB_sentence_L.append((textID, 句子清洗(line[2]).split()))
        return candTextF_sentence_L, candTextB_sentence_L, testTextF_sentence_L, testTextB_sentence_L

    def get一个向量(self):
        senID, word_L = self._candTextB_sentence_L[100]
        print('------new')
        mid = self._newH5['mid'][self._newH5_senID_no_D[senID]]
        matrix = self._newH5['matrix'][self._newH5_senID_no_D[senID]]
        print(matrix[mid:])
        print(self._newH5['length'][self._newH5_senID_no_D[senID]])
        print(mid)
        print(senID)
        print('------old')
        print(self._oldH5['matrixs'][self._oldH5_senID_no_D[' '.join(word_L)]])
        print(self._oldH5['lengths'][self._oldH5_senID_no_D[' '.join(word_L)]])
        print(word_L)


if __name__ == '__main__':
    比较不同h5py_obj = 比较不同h5py(oldH5=r"F:\data\_large_tem_file\CTE\data\CR dblp\av_xlnet-large-cased_sen_mat_len.h5",
                            newH5=r"F:\data\_large_tem_file\CTE\data\CR dblp\bc_xlnet-large-cased_senID_mat_len_seg_mid.h5",
                            datasetPath=r'D:\data\code\python\paper\text-representation\data\CR dblp\dataset.text')
    比较不同h5py_obj.get一个向量()
