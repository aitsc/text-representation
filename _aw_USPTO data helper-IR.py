import pickle
from tqdm import tqdm
import random
import sys
import os
import re
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 提取关键信息:
    def __init__(self, patentPath=None, ipcrPath=None, claimPath=None, 已提取信息地址=None):
        if 已提取信息地址:
            print('从地址中读取信息...')
            with open(已提取信息地址.encode('utf-8'), 'rb') as r:
                self._id_date_title_abstract_claim_class_D = pickle.load(r)
            print('专利数量: %d' % (len(self._id_date_title_abstract_claim_class_D)))
        else:
            id_date_title_abstract_claim_class_D = {}  # {id:[date, title, abstract, claim, {class,..}],..}
            # 提取 id/标题/摘要/年份
            有效专利 = 0
            with open(patentPath, 'r', encoding='utf-8') as r:
                r.readline()  # ['id', 'type', 'number', 'country', 'date', 'abstract', 'title', 'kind', 'num_claims', 'filename', 'withdrawn']
                for i, line in tqdm(enumerate(r), '提取 patent.tsv'):
                    line = line.strip('\r\n').split('\t')
                    if len(line) != 11:
                        continue
                    patent_id = line[0].strip()
                    year = line[4].strip().split('-')[0]
                    abstract = line[5].strip().lower()
                    title = line[6].strip().lower()
                    if not patent_id or not year or not abstract or not title:
                        continue
                    date_title_abstract_claim_class = [None, None, None, [], set()]
                    date_title_abstract_claim_class[1] = title
                    date_title_abstract_claim_class[2] = abstract
                    date_title_abstract_claim_class[0] = year
                    id_date_title_abstract_claim_class_D[patent_id] = date_title_abstract_claim_class
                    有效专利 += 1
            print('提取有效专利:%d, 字典中专利数:%d' % (有效专利, len(id_date_title_abstract_claim_class_D)))
            # 提取 分类
            有效专利 = 0
            with open(ipcrPath, 'r', encoding='utf-8') as r:
                r.readline()  # ['uuid', 'patent_id', 'classification_level', 'section', 'ipc_class', 'subclass', 'main_group', 'subgroup', 'symbol_position', 'classification_value', 'classification_status', 'classification_data_source', 'action_date', 'ipc_version_indicator', 'sequence']
                for i, line in tqdm(enumerate(r), '提取 ipcr.tsv'):
                    line = line.strip('\r\n').split('\t')
                    if len(line) != 15:
                        continue
                    patent_id = line[1].strip()
                    if patent_id not in id_date_title_abstract_claim_class_D:
                        continue
                    section = line[3].strip()
                    ipc_class = line[4].strip()
                    subclass = line[5].strip()
                    main_group = line[6].strip()
                    subgroup = line[7].strip()
                    if not patent_id or not section or not ipc_class or not subclass or not main_group or not subgroup:
                        continue
                    c = '-'.join([section, ipc_class, subclass, main_group, subgroup])
                    id_date_title_abstract_claim_class_D[patent_id][4].add(c)
                    有效专利 += 1
            print('提取有效行数:%d' % 有效专利)
            # 提取 claim
            有效专利 = 0
            with open(claimPath, 'r', encoding='utf-8') as r:
                r.readline()  # ['uuid', 'patent_id', 'text', 'dependent', 'sequence', 'exemplary']
                for i, line in tqdm(enumerate(r), '提取 claim.tsv'):
                    line = line.strip('\r\n').split('\t')
                    if len(line) != 6:
                        continue
                    patent_id = line[1].strip()
                    if patent_id not in id_date_title_abstract_claim_class_D:
                        continue
                    claim = line[2].strip().lower()
                    sequence = int(line[4].strip())  # 表示第几段claim
                    exemplary = line[5].strip().lower()  # 是否为示范权力要求之一, true or false
                    # 可以根据dependent和exemplary筛选claim
                    if sequence != 1:
                        continue
                    if not patent_id or not claim:
                        continue
                    id_date_title_abstract_claim_class_D[patent_id][3].append((sequence, claim))
                    有效专利 += 1
            print('提取有效行数:%d' % 有效专利)
            # 剔除有空属性的专利
            delete = []
            for i, date_title_abstract_claim_class in tqdm(id_date_title_abstract_claim_class_D.items(), '整理专利属性'):
                claim = date_title_abstract_claim_class[3]
                classification = date_title_abstract_claim_class[4]
                if not claim or not classification:
                    delete.append(i)
                claim = sorted(claim, key=lambda t: t[0])  # 按句子位置排序
                claim = ' '.join([i for _, i in claim])  # 去掉序号, 并合并
                date_title_abstract_claim_class[3] = claim
            for i in delete:
                del id_date_title_abstract_claim_class_D[i]
            print('最终提取到%d篇全属性专利' % len(id_date_title_abstract_claim_class_D))
            self._id_date_title_abstract_claim_class_D = id_date_title_abstract_claim_class_D

    def statistics(self):
        titleLen_max_min_avg = [0, 10**10, 0.]
        abstractLen_max_min_avg = [0, 10**10, 0.]
        claimLen_max_min_avg = [0, 10**10, 0.]
        patentClassNum_max_min_avg = [0, 10**10, 0.]
        yearPatentNum_D = {}
        class_num_D = {}
        for i, date_title_abstract_claim_class in tqdm(self._id_date_title_abstract_claim_class_D.items(), '数据统计'):
            year = date_title_abstract_claim_class[0]
            titleLen = len(句子清洗(date_title_abstract_claim_class[1]).split())
            abstractLen = len(句子清洗(date_title_abstract_claim_class[2]).split())
            claimLen = len(句子清洗(date_title_abstract_claim_class[3]).split())
            class_S = date_title_abstract_claim_class[4]

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

            if claimLen > claimLen_max_min_avg[0]:
                claimLen_max_min_avg[0] = claimLen
            if claimLen < claimLen_max_min_avg[1]:
                claimLen_max_min_avg[1] = claimLen
            claimLen_max_min_avg[2] += claimLen

            if len(class_S) > patentClassNum_max_min_avg[0]:
                patentClassNum_max_min_avg[0] = len(class_S)
            if len(class_S) < patentClassNum_max_min_avg[1]:
                patentClassNum_max_min_avg[1] = len(class_S)
            patentClassNum_max_min_avg[2] += len(class_S)

            for classification in class_S:
                if classification in class_num_D:
                    class_num_D[classification] += 1
                else:
                    class_num_D[classification] = 1

            if year in yearPatentNum_D:
                yearPatentNum_D[year] += 1
            else:
                yearPatentNum_D[year] = 1
        titleLen_max_min_avg[2] /= len(self._id_date_title_abstract_claim_class_D)
        abstractLen_max_min_avg[2] /= len(self._id_date_title_abstract_claim_class_D)
        claimLen_max_min_avg[2] /= len(self._id_date_title_abstract_claim_class_D)
        patentClassNum_max_min_avg[2] /= len(self._id_date_title_abstract_claim_class_D)
        classNum_L = [i for _, i in class_num_D.items()]
        分类拥有专利数_max_min_avg = [max(classNum_L), min(classNum_L), sum(classNum_L)/len(classNum_L)]
        print('titleLen_max_min_avg: %s, abstractLen_max_min_avg: %s, claimLen_max_min_avg: %s, patentClassNum_max_min_avg: %s, 分类拥有专利数_max_min_avg: %s, 分类总数: %d' %
              (str(titleLen_max_min_avg), str(abstractLen_max_min_avg), str(claimLen_max_min_avg), str(patentClassNum_max_min_avg), str(分类拥有专利数_max_min_avg), len(class_num_D)))
        print('年份_专利数: %s' % str(sorted(yearPatentNum_D.items(), key=lambda t: int(t[0]), reverse=True)))

    def saveFile(self, path):
        print('保存,二进制化...')
        二进制 = pickle.dumps(self._id_date_title_abstract_claim_class_D)
        缓存 = 10 ** 6
        with open(path.encode('utf-8'), 'wb') as w:
            for i in tqdm(range(0, len(二进制), 缓存), '保存 id_date_title_abstract_claim_class_D'):
                w.write(二进制[i:i + 缓存])

    def writeCorpus(self, path, textFilter=None):
        '''
        没有过滤器则输出纯粹的预料(双倍), 输出2遍, 第一遍原文, 第二遍将非字母数字替换为空格, 用于词向量训练
        '''
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_date_title_abstract_claim_class_D, '没有专利信息!'
        with open(path, 'w', encoding='utf-8') as w:
            if not textFilter:
                for _, (_, title, abstract, claim, _) in tqdm(self._id_date_title_abstract_claim_class_D.items(), mName+'输出第一遍'):
                    w.write(title + '\n')
                    w.write(abstract + '\n')
                    w.write(claim + '\n')
                w.write('\n')
                for _, (_, title, abstract, claim, _) in tqdm(self._id_date_title_abstract_claim_class_D.items(), mName+'输出第二遍'):
                    w.write(re.sub('[^0-9a-zA-Z]+', ' ', title) + '\n')
                    w.write(re.sub('[^0-9a-zA-Z]+', ' ', abstract) + '\n')
                    w.write(re.sub('[^0-9a-zA-Z]+', ' ', claim) + '\n')
            else:
                for _, (_, title, abstract, claim, _) in tqdm(self._id_date_title_abstract_claim_class_D.items(), mName+'输出'):
                    w.write(textFilter(title) + '\n')
                    w.write(textFilter(abstract) + '\n')
                    w.write(textFilter(claim) + '\n')

    @property
    def id_date_title_abstract_claim_class_D(self):
        return self._id_date_title_abstract_claim_class_D


class PatentFilter:  # paper复用的过滤代码
    def __init__(self, id_title_author_c1_c2_abstract_year_L):
        # patent: [(id,title+abstract,[class,..],None,claim,year),..]
        self._id_title_author_c1_c2_abstract_year_D = {i[0]: i[1:] for i in id_title_author_c1_c2_abstract_year_L}
        self._trainPaper_wordNum_year_classNum_L = {}  # [(paperID,单词数量,年份,类别数量),..]
        self._testPaper_labelS_wordNum_year_classNum_L = {}  # {paperID:(检索到的专利set,单词数量,年份,类别数量),..}

    def filter(self, trainPaperNum=10000, trainPaperYear_T=(0, 2015), paperTitleWordNum_T=(5, 50),
               testPaperNum=100, testPaperYear_T=(2016, 2019), paperAbstractWordNum_T=(150, 1000),
               minRetrievalPaperNum=20, orderChoiceTrainPaper=True):
        '''
        先选择训练集专利, 然后计算分类对应训练集专利用于限定测试集专利的标准答案数量, 然后选择测试集专利
        '''
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        whatClassUsed = 2

        # 筛选满足条件的训练集专利
        trainPaper_wordNum_year_classNum_L = []  # [(paperID,单词数量,年份,类别数量),..]
        for paperId, paperInfor_L in tqdm(self._id_title_author_c1_c2_abstract_year_D.items(), mName+'-筛选训练集专利'):
            title = paperInfor_L[0]
            abstract = paperInfor_L[4]
            year = paperInfor_L[5]
            text = title + ' ' + abstract
            if not trainPaperYear_T[0] <= year <= trainPaperYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not paperTitleWordNum_T[0] <= len(title.split()) <= paperTitleWordNum_T[1]:  # 专利title单词数要求
                continue
            if not paperAbstractWordNum_T[0] <= len(abstract.split()) <= paperAbstractWordNum_T[1]:  # 专利abstract单词数要求
                continue
            trainPaper_wordNum_year_classNum_L.append((paperId, wordNum, year, len(paperInfor_L[whatClassUsed])))
        trainPaperNumAll = len(trainPaper_wordNum_year_classNum_L)  # 所有满足条件的训练集的专利数量
        if orderChoiceTrainPaper or trainPaperNum >= len(trainPaper_wordNum_year_classNum_L):
            trainPaper_wordNum_year_classNum_L = trainPaper_wordNum_year_classNum_L[:trainPaperNum]
        else:
            trainPaper_wordNum_year_classNum_L = random.sample(trainPaper_wordNum_year_classNum_L, trainPaperNum)
        trainPaper_S = set([i[0] for i in trainPaper_wordNum_year_classNum_L])

        # 计算每类别专利
        class_testPaper_D = {}  # {class:{paperID,..},..}
        for paper, _, _, _ in trainPaper_wordNum_year_classNum_L:
            class_L = self._id_title_author_c1_c2_abstract_year_D[paper][whatClassUsed]
            for c in class_L:
                if c in class_testPaper_D:
                    class_testPaper_D[c].add(paper)
                else:
                    class_testPaper_D[c] = {paper}

        # 筛选满足条件的测试集专利
        testPaper_labelS_wordNum_year_classNum_D = {}  # {paperID:(检索到的专利set,单词数量,年份,类别数量),..}
        alltestPaperClass_S = set()
        alltestPaperClassAll_S = set()  # 包含没有在训练集中出现的分类
        for paperId, paperInfor_L in tqdm(self._id_title_author_c1_c2_abstract_year_D.items(), mName+'-筛选测试集专利'):
            if paperId in trainPaper_S:  # 不能是训练集专利
                continue
            title = paperInfor_L[0]
            abstract = paperInfor_L[4]
            year = paperInfor_L[5]
            text = title + ' ' + abstract
            if not testPaperYear_T[0] <= year <= testPaperYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not paperTitleWordNum_T[0] <= len(title.split()) <= paperTitleWordNum_T[1]:  # 专利title单词数要求
                continue
            if not paperAbstractWordNum_T[0] <= len(abstract.split()) <= paperAbstractWordNum_T[1]:  # 专利abstract单词数要求
                continue
            labelPaper_S = set()
            testPaperClassL = []
            for c in paperInfor_L[whatClassUsed]:
                if c in class_testPaper_D:
                    labelPaper_S |= class_testPaper_D[c]
                    testPaperClassL.append(c)
            if minRetrievalPaperNum > len(labelPaper_S):  # 满足测试专利最少检索到的专利数量
                continue
            testPaper_labelS_wordNum_year_classNum_D[paperId] = (labelPaper_S, wordNum, year, len(testPaperClassL))
            alltestPaperClass_S |= set(testPaperClassL)
            alltestPaperClassAll_S |= set(paperInfor_L[whatClassUsed])
        testPaperNumAll = len(testPaper_labelS_wordNum_year_classNum_D)  # 所有满足条件的测试集专利数量
        testPaper_labelS_wordNum_year_classNum_L = sorted(testPaper_labelS_wordNum_year_classNum_D.items(), key=lambda t: len(t[1][0]))
        testPaperMaxLabelNumAll = len(testPaper_labelS_wordNum_year_classNum_L[-1][1][0])  # 所有满足条件的测试集专利中标签最多的专利的标签数量
        testPaper_labelS_wordNum_year_classNum_L = testPaper_labelS_wordNum_year_classNum_L[:testPaperNum]

        # 统计信息
        trainPaperYearMaxMinAve_L = [0, 1000000, 0]  # 训练集专利年份
        trainPaperWordNumMaxMinAve_L = [0, 1000000, 0]
        trainClassNumMaxMinAve_L = [0, 1000000, 0]
        classTrainPaperNumMaxMinAve_L = [0, 1000000, 0]  # 测试集所有专利的分类的测试集专利数量

        testPaperYearMaxMinAve_L = [0, 1000000, 0]
        testPaperWordMaxMinAve_L = [0, 1000000, 0]
        testPaperLabelNumMaxMinAve_L = [0, 1000000, 0]
        testPaperClassNumMaxMinAve_L = [0, 1000000, 0]
        # 训练集专利统计信息
        for _, wordNum, year, classNum in tqdm(trainPaper_wordNum_year_classNum_L, mName+'-统计训练集专利信息'):
            # 专利年份
            if trainPaperYearMaxMinAve_L[0] < year:
                trainPaperYearMaxMinAve_L[0] = year
            if trainPaperYearMaxMinAve_L[1] > year:
                trainPaperYearMaxMinAve_L[1] = year
            trainPaperYearMaxMinAve_L[2] += year
            # 专利词数
            if trainPaperWordNumMaxMinAve_L[0] < wordNum:
                trainPaperWordNumMaxMinAve_L[0] = wordNum
            if trainPaperWordNumMaxMinAve_L[1] > wordNum:
                trainPaperWordNumMaxMinAve_L[1] = wordNum
            trainPaperWordNumMaxMinAve_L[2] += wordNum
            # 分类数
            if trainClassNumMaxMinAve_L[0] < classNum:
                trainClassNumMaxMinAve_L[0] = classNum
            if trainClassNumMaxMinAve_L[1] > classNum:
                trainClassNumMaxMinAve_L[1] = classNum
            trainClassNumMaxMinAve_L[2] += classNum
        trainPaperYearMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        trainPaperWordNumMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        trainClassNumMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        # 测试集所有专利的分类的测试集专利数量
        for _, s in class_testPaper_D.items():
            if classTrainPaperNumMaxMinAve_L[0] < len(s):
                classTrainPaperNumMaxMinAve_L[0] = len(s)
            if classTrainPaperNumMaxMinAve_L[1] > len(s):
                classTrainPaperNumMaxMinAve_L[1] = len(s)
            classTrainPaperNumMaxMinAve_L[2] += len(s)
        classTrainPaperNumMaxMinAve_L[2] /= len(class_testPaper_D)
        # 测试集统计信息
        for _, v in tqdm(testPaper_labelS_wordNum_year_classNum_L, mName+'-统计测试集专利信息'):
            labelS, wordNum, year, classNum = v
            # 专利年份
            if testPaperYearMaxMinAve_L[0] < year:
                testPaperYearMaxMinAve_L[0] = year
            if testPaperYearMaxMinAve_L[1] > year:
                testPaperYearMaxMinAve_L[1] = year
            testPaperYearMaxMinAve_L[2] += year
            # 专利词数
            if testPaperWordMaxMinAve_L[0] < wordNum:
                testPaperWordMaxMinAve_L[0] = wordNum
            if testPaperWordMaxMinAve_L[1] > wordNum:
                testPaperWordMaxMinAve_L[1] = wordNum
            testPaperWordMaxMinAve_L[2] += wordNum
            # 专利检索到的专利数
            if testPaperLabelNumMaxMinAve_L[0] < len(labelS):
                testPaperLabelNumMaxMinAve_L[0] = len(labelS)
            if testPaperLabelNumMaxMinAve_L[1] > len(labelS):
                testPaperLabelNumMaxMinAve_L[1] = len(labelS)
            testPaperLabelNumMaxMinAve_L[2] += len(labelS)
            # 分类数
            if testPaperClassNumMaxMinAve_L[0] < classNum:
                testPaperClassNumMaxMinAve_L[0] = classNum
            if testPaperClassNumMaxMinAve_L[1] > classNum:
                testPaperClassNumMaxMinAve_L[1] = classNum
            testPaperClassNumMaxMinAve_L[2] += classNum
        testPaperYearMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_classNum_L)
        testPaperWordMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_classNum_L)
        testPaperLabelNumMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_classNum_L)
        testPaperClassNumMaxMinAve_L[2] /= len(testPaper_labelS_wordNum_year_classNum_L)

        print('满足条件的理论训练集专利数:%d, 满足条件的理论测试集专利数:%d, 理论测试集专利最大标签数:%d' % (trainPaperNumAll, testPaperNumAll, testPaperMaxLabelNumAll))
        print('训练集专利MaxMinAve年份:%s, 训练集专利MaxMinAve词数:%s, 训练集专利MaxMinAve分类:%s, '
              '训练集专利总数:%d, 训练集分类种数:%d, 分类MaxMinAve训练集专利数:%s' %
              (str(trainPaperYearMaxMinAve_L), str(trainPaperWordNumMaxMinAve_L), str(trainClassNumMaxMinAve_L),
               len(trainPaper_wordNum_year_classNum_L), len(class_testPaper_D), str(classTrainPaperNumMaxMinAve_L)))
        print('测试集专利MaxMinAve年份:%s, 测试集专利MaxMinAve词数:%s, 测试集专利MaxMinAve标准答案:%s, 测试集专利MaxMinAve分类:%s, '
              '测试集专利数:%d, 测试集专利分类种数:%d, 测试集专利分类种数(包含没有在训练集中出现的分类):%d' %
              (str(testPaperYearMaxMinAve_L), str(testPaperWordMaxMinAve_L), str(testPaperLabelNumMaxMinAve_L), str(testPaperClassNumMaxMinAve_L),
               len(testPaper_labelS_wordNum_year_classNum_L), len(alltestPaperClass_S), len(alltestPaperClassAll_S)))

        self._trainPaper_wordNum_year_classNum_L = trainPaper_wordNum_year_classNum_L
        self._testPaper_labelS_wordNum_year_classNum_L = testPaper_labelS_wordNum_year_classNum_L

    def savePaperFolder(self, path, labelName='测试专利编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=False):
        assert self._trainPaper_wordNum_year_classNum_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_classNum_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path), '删除原文件夹内容...'):
                    os.remove(path + '/' + i)
        for paperID, _, _, _ in tqdm(self._trainPaper_wordNum_year_classNum_L, mName+'-写入训练集专利文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(text)
        for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName+'-写入测试集专利文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(text)
        with open(path + '/' + labelName, 'w', encoding='utf-8') as w:
            for paperID, v in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName+'-写入测试集专利标签(标准答案)'):
                labelS = v[0]
                w.write(paperID+'\t'+'\t'.join(labelS)+'\n')

    def savePaperFile(self, address, segTitleAbstract='\t'):
        assert self._trainPaper_wordNum_year_classNum_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_classNum_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        testPaper_label_D = {i: v[0] for i, v in self._testPaper_labelS_wordNum_year_classNum_L}  # {测试集专利:训练集专利答案set,..}
        with open(address, 'w', encoding='utf-8') as w:
            w.write(str(testPaper_label_D)+'\n')
            for paperID, _, _, _ in tqdm(self._trainPaper_wordNum_year_classNum_L, mName + '-写入训练集专利文本'):
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(paperID + '\t' + text + '\n')
            for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName + '-写入测试集专利文本'):
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(paperID + '\t' + text + '\n')


if __name__ == '__main__':
    # 提取关键信息_obj = 提取关键信息(patentPath=r'F:\data\USPTO\patent.tsv',
    #                     ipcrPath=r'F:\data\USPTO\ipcr.tsv',
    #                     claimPath=r'F:\data\USPTO\claim.tsv')
    # 提取关键信息_obj.saveFile(r'data\all USPTO\aw_USPTO-id_date_title_abstract_claim_class_D.pkl')
    提取关键信息_obj = 提取关键信息(已提取信息地址=r'data/all USPTO/aw_USPTO-id_date_title_abstract_claim_class_D.pkl')
    提取关键信息_obj.writeCorpus(path=r'data/all USPTO/aw_USPTOCorpus.text', textFilter=句子清洗)
    # 提取关键信息_obj.statistics()
    
    # 构建PatentFilter的输入变量
    id_title_author_c1_c2_abstract_year_L = []  # [(id,None,title+abstract,[class,..],None,claim,year),..]
    for i, date_title_abstract_claim_class in 提取关键信息_obj.id_date_title_abstract_claim_class_D.items():
        year = int(date_title_abstract_claim_class[0])
        title = date_title_abstract_claim_class[1] + '. ' + date_title_abstract_claim_class[2]
        abstract = date_title_abstract_claim_class[3]
        c_L = list(date_title_abstract_claim_class[4])
        id_title_author_c1_c2_abstract_year_L.append((i, title, None, c_L, None, abstract, year))

    # 开始过滤
    PatentFilter_obj = PatentFilter(id_title_author_c1_c2_abstract_year_L)
    PatentFilter_obj.filter(trainPaperNum=20000, trainPaperYear_T=(0, 2018), paperTitleWordNum_T=(100, 300),
                            testPaperNum=1000, testPaperYear_T=(2019, 2019), paperAbstractWordNum_T=(150, 400),
                            minRetrievalPaperNum=20, orderChoiceTrainPaper=False)
    print('按任意键继续保存文件...')
    os.system("pause")  # linux 不暂停
    # PatentFilter_obj.savePaperFolder(path='data/IR USPTO/dataset/', labelName='测试专利编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=True)
    PatentFilter_obj.savePaperFile(address='data/IR USPTO/dataset.text', segTitleAbstract='\t')
