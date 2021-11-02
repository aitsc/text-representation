import pickle
from tqdm import tqdm
import random
import sys
import os
import re
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 提取关键信息:
    def __init__(self, paperPath=(), 已提取信息地址=None):
        if 已提取信息地址:
            print('从地址中读取信息...')
            with open(已提取信息地址.encode('utf-8'), 'rb') as r:
                id_year_title_abstract_ref_D = pickle.load(r)
            print('论文数量: %d' % (len(id_year_title_abstract_ref_D)))
        else:
            id_year_title_abstract_ref_D = {}  # {id:[year,title,abstract,{ref,..}],..}
            for path in paperPath:
                with open(path, 'r', encoding='utf-8') as r:
                    for i, line in tqdm(enumerate(r), '读取 '+path):
                        line = eval(line.strip())
                        try:
                            paper_id = line['id']
                            year = line['year']
                            title = line['title'].strip().lower()
                            abstract = line['abstract'].strip().lower()
                            references = line['references']
                        except:
                            continue
                        if not paper_id or not year or not title or not abstract or not references:
                            continue
                        year_title_abstract_ref = [None]*4
                        year_title_abstract_ref[0] = year
                        year_title_abstract_ref[1] = title
                        year_title_abstract_ref[2] = abstract
                        year_title_abstract_ref[3] = references
                        id_year_title_abstract_ref_D[paper_id] = year_title_abstract_ref
            # 过滤不在论文中的引文
            过滤的引文数量 = 0
            for _, year_title_abstract_ref in tqdm(id_year_title_abstract_ref_D.items(), '过滤不在论文中的引文'):
                references = year_title_abstract_ref[3]
                notDel = []
                for ref in references:
                    if ref in id_year_title_abstract_ref_D:
                        notDel.append(ref)
                if len(notDel) < len(references):
                    year_title_abstract_ref[3] = notDel
                    过滤的引文数量 += len(references) - len(notDel)
            print('最终提取到%d篇全属性论文, 过滤的引文数量:%d' % (len(id_year_title_abstract_ref_D), 过滤的引文数量))
        self._id_year_title_abstract_ref_D = id_year_title_abstract_ref_D

    def statistics(self):
        titleLen_max_min_avg = [0, 10**10, 0.]
        abstractLen_max_min_avg = [0, 10**10, 0.]
        paperRefNum_max_min_avg = [0, 10**10, 0.]
        yearPaperNum_D = {}
        for i, year_title_abstract_ref in tqdm(self._id_year_title_abstract_ref_D.items(), '数据统计'):
            year = year_title_abstract_ref[0]
            titleLen = len(句子清洗(year_title_abstract_ref[1]).split())
            abstractLen = len(句子清洗(year_title_abstract_ref[2]).split())
            references = year_title_abstract_ref[3]

            if titleLen > titleLen_max_min_avg[0]:
                titleLen_max_min_avg[0] = titleLen
            if titleLen < titleLen_max_min_avg[1]:
                titleLen_max_min_avg[1] = titleLen
            titleLen_max_min_avg[2] += titleLen

            if abstractLen > abstractLen_max_min_avg[0]:
                abstractLen_max_min_avg[0] = abstractLen
            if abstractLen < abstractLen_max_min_avg[1]:
                abstractLen_max_min_avg[1] = abstractLen
            abstractLen_max_min_avg[2] += abstractLen

            if len(references) > paperRefNum_max_min_avg[0]:
                paperRefNum_max_min_avg[0] = len(references)
            if len(references) < paperRefNum_max_min_avg[1]:
                paperRefNum_max_min_avg[1] = len(references)
            paperRefNum_max_min_avg[2] += len(references)

            if year in yearPaperNum_D:
                yearPaperNum_D[year] += 1
            else:
                yearPaperNum_D[year] = 1
        titleLen_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        abstractLen_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        paperRefNum_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        print('titleLen_max_min_avg: %s, abstractLen_max_min_avg: %s, paperRefNum_max_min_avg: %s' %
              (str(titleLen_max_min_avg), str(abstractLen_max_min_avg), str(paperRefNum_max_min_avg)))
        print('年份_论文数: %s' % str(sorted(yearPaperNum_D.items(), key=lambda t: int(t[0]), reverse=True)))

    def saveFile(self, path):
        print('保存,二进制化...')
        二进制 = pickle.dumps(self._id_year_title_abstract_ref_D)
        缓存 = 10 ** 6
        with open(path.encode('utf-8'), 'wb') as w:
            for i in tqdm(range(0, len(二进制), 缓存), '保存 id_year_title_abstract_ref_D'):
                w.write(二进制[i:i + 缓存])

    def writeCorpus(self, path, textFilter=None):
        '''
        没有过滤器则输出纯粹的预料(双倍), 输出2遍, 第一遍原文, 第二遍将非字母数字替换为空格, 用于词向量训练
        '''
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_year_title_abstract_ref_D, '没有论文信息!'
        with open(path, 'w', encoding='utf-8', errors='ignore') as w:
            if not textFilter:
                for _, (_, title, abstract, _) in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'输出第一遍'):
                    w.write(title + '\n')
                    w.write(abstract + '\n')
                w.write('\n')
                for _, (_, title, abstract, _) in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'输出第二遍'):
                    w.write(re.sub('[^0-9a-zA-Z]+', ' ', title) + '\n')
                    w.write(re.sub('[^0-9a-zA-Z]+', ' ', abstract) + '\n')
            else:
                for _, (_, title, abstract, _) in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'输出'):
                    w.write(textFilter(title) + '\n')
                    w.write(textFilter(abstract) + '\n')

    @property
    def id_year_title_abstract_ref_D(self):
        return self._id_year_title_abstract_ref_D


class PaperFilter:  # 这里的词数量统计不使用句子清洗
    def __init__(self, id_year_title_abstract_ref_D):
        self._id_year_title_abstract_ref_D = id_year_title_abstract_ref_D  # {id:[year,title,abstract,{ref,..}],..}
        self._trainPaper_wordNum_year_L = {}  # [(paperID,单词数量,年份),..]
        self._testPaper_labelS_wordNum_year_L = {}  # {paperID:(引文set,单词数量,年份),..}

    def filter(self, trainPaperNum=10000, trainPaperYear_T=(0, 2015), paperTitleWordNum_T=(5, 50),
               testPaperNum=100, testPaperYear_T=(2016, 2019), paperAbstractWordNum_T=(150, 1000),
               minRefPaperNum=20, orderChoiceTrainPaper=True):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name

        # 筛选满足条件的训练集论文, 保证提出的论文都是满足单词要求的
        trainPaper_wordNum_year_L = []  # [(paperID,单词数量,年份),..]
        for i, year_title_abstract_ref in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'-筛选训练集论文'):
            year = year_title_abstract_ref[0]
            title = year_title_abstract_ref[1]
            abstract = year_title_abstract_ref[2]
            text = title + ' ' + abstract
            if not trainPaperYear_T[0] <= year <= trainPaperYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not paperTitleWordNum_T[0] <= len(title.split()) <= paperTitleWordNum_T[1]:  # 论文title单词数要求
                continue
            if not paperAbstractWordNum_T[0] <= len(abstract.split()) <= paperAbstractWordNum_T[1]:  # 论文abstract单词数要求
                continue
            trainPaper_wordNum_year_L.append((i, wordNum, year))
        trainPaperNumAll = len(trainPaper_wordNum_year_L)  # 所有满足条件的训练集的论文数量
        trainPaper_S = set([i[0] for i in trainPaper_wordNum_year_L])  # 用于筛选训练集引文

        # 筛选满足条件的测试集论文, 去除不在以上训练集中的引文
        testPaper_labelS_wordNum_year_D = {}  # {paperID:(引文set,单词数量,年份),..}
        for i, year_title_abstract_ref in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'-筛选测试集论文'):
            if i in trainPaper_S:  # 不能是训练集论文
                continue
            year = year_title_abstract_ref[0]
            title = year_title_abstract_ref[1]
            abstract = year_title_abstract_ref[2]
            references = year_title_abstract_ref[3]  # list
            text = title + ' ' + abstract
            if not testPaperYear_T[0] <= year <= testPaperYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not paperTitleWordNum_T[0] <= len(title.split()) <= paperTitleWordNum_T[1]:  # 论文title单词数要求
                continue
            if not paperAbstractWordNum_T[0] <= len(abstract.split()) <= paperAbstractWordNum_T[1]:  # 论文abstract单词数要求
                continue
            labelPaper_S = set(references) & trainPaper_S  # 必须在训练集中
            if minRefPaperNum > len(labelPaper_S):  # 满足测试论文最少检索到的论文数量
                continue
            testPaper_labelS_wordNum_year_D[i] = (labelPaper_S, wordNum, year)
        testPaperNumAll = len(testPaper_labelS_wordNum_year_D)  # 所有满足条件的测试集论文数量
        testPaper_labelS_wordNum_year_L = sorted(testPaper_labelS_wordNum_year_D.items(), key=lambda t: len(t[1][0]))
        testPaperMaxLabelNumAll = len(testPaper_labelS_wordNum_year_L[-1][1][0])  # 所有满足条件的测试集论文中标签最多的论文的标签数量
        testPaper_labelS_wordNum_year_L = testPaper_labelS_wordNum_year_L[:testPaperNum]
        allRef_S = set()  # 用于筛选训练集
        for _, (ref_S, _, _) in testPaper_labelS_wordNum_year_L:
            allRef_S |= ref_S

        # 获得真正的训练集, 合并引文,然后加入剩下需要的论文
        trainPaper_L = list(allRef_S)
        if orderChoiceTrainPaper:  # 不随机
            i = 0
            while len(trainPaper_L) < trainPaperNum:
                j = trainPaper_wordNum_year_L[i][0]
                if j not in allRef_S:
                    trainPaper_L.append(j)
                i += 1
        else:  # 随机挑选
            if trainPaperNum > len(trainPaper_L):
                剩余论文 = list(trainPaper_S - allRef_S)
                random.shuffle(剩余论文)
                trainPaper_L += 剩余论文[:trainPaperNum-len(trainPaper_L)]
        trainPaper_S = set(trainPaper_L)  # 新的训练集论文id集合
        trainPaper_wordNum_year_L_real = []
        for i, wordNum, year in trainPaper_wordNum_year_L:  # 真正的训练集挑选出来
            if i in trainPaper_S:
                trainPaper_wordNum_year_L_real.append((i, wordNum, year))
        trainPaper_wordNum_year_L = trainPaper_wordNum_year_L_real

        # 统计信息
        trainPaperYearMaxMinAve_L = [0, 1000000, 0]  # 训练集论文年份
        trainPaperWordNumMaxMinAve_L = [0, 1000000, 0]

        testPaperYearMaxMinAve_L = [0, 1000000, 0]
        testPaperWordMaxMinAve_L = [0, 1000000, 0]
        testPaperLabelNumMaxMinAve_L = [0, 1000000, 0]
        # 训练集论文统计信息
        for _, wordNum, year in tqdm(trainPaper_wordNum_year_L, mName+'-统计训练集论文信息'):
            # 论文年份
            if trainPaperYearMaxMinAve_L[0] < year:
                trainPaperYearMaxMinAve_L[0] = year
            if trainPaperYearMaxMinAve_L[1] > year:
                trainPaperYearMaxMinAve_L[1] = year
            trainPaperYearMaxMinAve_L[2] += year
            # 论文词数
            if trainPaperWordNumMaxMinAve_L[0] < wordNum:
                trainPaperWordNumMaxMinAve_L[0] = wordNum
            if trainPaperWordNumMaxMinAve_L[1] > wordNum:
                trainPaperWordNumMaxMinAve_L[1] = wordNum
            trainPaperWordNumMaxMinAve_L[2] += wordNum
        trainPaperYearMaxMinAve_L[2] /= len(trainPaper_wordNum_year_L)
        trainPaperWordNumMaxMinAve_L[2] /= len(trainPaper_wordNum_year_L)
        # 测试集统计信息
        for _, v in tqdm(testPaper_labelS_wordNum_year_L, mName+'-统计测试集论文信息'):
            labelS, wordNum, year = v
            # 论文年份
            if testPaperYearMaxMinAve_L[0] < year:
                testPaperYearMaxMinAve_L[0] = year
            if testPaperYearMaxMinAve_L[1] > year:
                testPaperYearMaxMinAve_L[1] = year
            testPaperYearMaxMinAve_L[2] += year
            # 论文词数
            if testPaperWordMaxMinAve_L[0] < wordNum:
                testPaperWordMaxMinAve_L[0] = wordNum
            if testPaperWordMaxMinAve_L[1] > wordNum:
                testPaperWordMaxMinAve_L[1] = wordNum
            testPaperWordMaxMinAve_L[2] += wordNum
            # 引文数
            if testPaperLabelNumMaxMinAve_L[0] < len(labelS):
                testPaperLabelNumMaxMinAve_L[0] = len(labelS)
            if testPaperLabelNumMaxMinAve_L[1] > len(labelS):
                testPaperLabelNumMaxMinAve_L[1] = len(labelS)
            testPaperLabelNumMaxMinAve_L[2] += len(labelS)
        testPaperYearMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_L)
        testPaperWordMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_L)
        testPaperLabelNumMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_L)

        print('满足条件的理论训练集论文数:%d, 满足条件的理论测试集论文数:%d, 理论测试集论文最大标签数:%d' % (trainPaperNumAll, testPaperNumAll, testPaperMaxLabelNumAll))
        print('训练集论文MaxMinAve年份:%s, 训练集论文MaxMinAve词数:%s, 训练集论文总数:%d' %
              (str(trainPaperYearMaxMinAve_L), str(trainPaperWordNumMaxMinAve_L), len(trainPaper_wordNum_year_L)))
        print('测试集论文MaxMinAve年份:%s, 测试集论文MaxMinAve词数:%s, 测试集论文MaxMinAve引文数:%s, 测试集论文数:%d' %
              (str(testPaperYearMaxMinAve_L), str(testPaperWordMaxMinAve_L), str(testPaperLabelNumMaxMinAve_L), len(testPaper_labelS_wordNum_year_L)))

        self._trainPaper_wordNum_year_L = trainPaper_wordNum_year_L
        self._testPaper_labelS_wordNum_year_L = testPaper_labelS_wordNum_year_L

    def savePaperFolder(self, path, labelName='测试论文编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=False):
        assert self._trainPaper_wordNum_year_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path), '删除原文件夹内容...'):
                    os.remove(path + '/' + i)
        for paperID, _, _ in tqdm(self._trainPaper_wordNum_year_L, mName+'-写入训练集论文文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_year_title_abstract_ref_D[paperID]
                text = paperInfor_L[1] + segTitleAbstract + paperInfor_L[2]
                w.write(text)
        for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_L, mName+'-写入测试集论文文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_year_title_abstract_ref_D[paperID]
                text = paperInfor_L[1] + segTitleAbstract + paperInfor_L[2]
                w.write(text)
        with open(path + '/' + labelName, 'w', encoding='utf-8') as w:
            for paperID, v in tqdm(self._testPaper_labelS_wordNum_year_L, mName+'-写入测试集论文标签(标准答案)'):
                labelS = v[0]
                w.write(paperID+'\t'+'\t'.join(labelS)+'\n')

    def savePaperFile(self, address, segTitleAbstract='\t'):
        assert self._trainPaper_wordNum_year_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        testPaper_label_D = {i: v[0] for i, v in self._testPaper_labelS_wordNum_year_L}  # {测试集论文:训练集论文答案set,..}
        with open(address, 'w', encoding='utf-8') as w:
            w.write(str(testPaper_label_D)+'\n')  # 出现utf-8写入错误可以考虑多选取几次,选到没有错误的文本
            for paperID, _, _ in tqdm(self._trainPaper_wordNum_year_L, mName + '-写入训练集论文文本'):
                paperInfor_L = self._id_year_title_abstract_ref_D[paperID]
                text = paperInfor_L[1] + segTitleAbstract + paperInfor_L[2]
                w.write(paperID + '\t' + text + '\n')
            for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_L, mName + '-写入测试集论文文本'):
                paperInfor_L = self._id_year_title_abstract_ref_D[paperID]
                text = paperInfor_L[1] + segTitleAbstract + paperInfor_L[2]
                w.write(paperID + '\t' + text + '\n')


if __name__ == '__main__':
    # 提取关键信息_obj = 提取关键信息(paperPath=[
    #     r"F:\data\dblp\dblp.v10\dblp-ref-0.json",
    #     r"F:\data\dblp\dblp.v10\dblp-ref-1.json",
    #     r"F:\data\dblp\dblp.v10\dblp-ref-2.json",
    #     r"F:\data\dblp\dblp.v10\dblp-ref-3.json",
    # ])
    # 提取关键信息_obj.saveFile(r'data\all dblp\ax_dblp-id_year_title_abstract_ref_D.pkl')
    提取关键信息_obj = 提取关键信息(已提取信息地址=r'data/all dblp/ax_dblp-id_year_title_abstract_ref_D.pkl')
    # 提取关键信息_obj.writeCorpus(path=r'data/all dblp/ax_dblpCorpus.text', textFilter=句子清洗)
    # 提取关键信息_obj.statistics()

    # 开始过滤
    PaperFilter_obj = PaperFilter(提取关键信息_obj.id_year_title_abstract_ref_D)
    PaperFilter_obj.filter(trainPaperNum=20000, trainPaperYear_T=(0, 2016), paperTitleWordNum_T=(5, 30),
                           testPaperNum=1000, testPaperYear_T=(2017, 2019), paperAbstractWordNum_T=(150, 400),
                           minRefPaperNum=15, orderChoiceTrainPaper=False)
    print('按任意键继续保存文件...')
    os.system("pause")  # linux 不暂停
    # PaperFilter_obj.savePaperFolder(path='data/CR dblp/dataset/', labelName='测试论文编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=True)
    PaperFilter_obj.savePaperFile(address='data/CR dblp/dataset.text', segTitleAbstract='\t')
