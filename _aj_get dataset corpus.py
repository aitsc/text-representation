from tqdm import tqdm
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


def getPaperInfor(dataIRaddress):
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


if __name__ == '__main__':
    数据集地址 = 'data/TC AGnews/dataset.text'
    语料保存地址 = 'data/TC AGnews/aj_dataset corpus.text'

    trainPaper_sentence_L, testPaper_sentence_L = getPaperInfor(dataIRaddress=数据集地址)
    with open(语料保存地址, 'w', encoding='utf-8') as w:
        for ID, text in trainPaper_sentence_L:
            w.write(text+'\n')
        for ID, text in testPaper_sentence_L:
            w.write(text+'\n')
