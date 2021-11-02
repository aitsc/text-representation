import pickle
import os
import re
from tqdm import tqdm
import sys
import random
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class PaperFilter:
    def __init__(self, id_title_author_c1_c2_abstract_year_L):
        self._id_title_author_c1_c2_abstract_year_D = {i[0]: i[1:] for i in id_title_author_c1_c2_abstract_year_L}
        self._trainPaper_wordNum_year_classNum_L = {}  # [(paperID,单词数量,年份,类别数量),..]
        self._testPaper_labelS_wordNum_year_classNum_L = {}  # {paperID:(检索到的论文set,单词数量,年份,类别数量),..}

    def filter(self, trainPaperNum=10000, trainPaperYear_T=(0, 2015), paperTitleWordNum_T=(5, 50),
               testPaperNum=100, testPaperYear_T=(2016, 2019), paperAbstractWordNum_T=(150, 1000),
               whatClassUsed='subject', minRetrievalPaperNum=20, orderChoiceTrainPaper=True):
        '''
        先选择训练集论文, 然后计算分类对应训练集论文用于限定测试集论文的标准答案数量, 然后选择测试集论文
        '''
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name

        if whatClassUsed == 'subject':
            whatClassUsed = 2
        elif whatClassUsed == 'MSC':
            whatClassUsed = 3
        else:
            assert False, '分类错误!'

        # 筛选满足条件的训练集论文
        trainPaper_wordNum_year_classNum_L = []  # [(paperID,单词数量,年份,类别数量),..]
        for paperId, paperInfor_L in tqdm(self._id_title_author_c1_c2_abstract_year_D.items(), mName+'-筛选训练集论文'):
            title = paperInfor_L[0]
            abstract = paperInfor_L[4]
            year = paperInfor_L[5]
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
            trainPaper_wordNum_year_classNum_L.append((paperId, wordNum, year, len(paperInfor_L[whatClassUsed])))
        trainPaperNumAll = len(trainPaper_wordNum_year_classNum_L)  # 所有满足条件的训练集的论文数量
        if orderChoiceTrainPaper or trainPaperNum >= len(trainPaper_wordNum_year_classNum_L):
            trainPaper_wordNum_year_classNum_L = trainPaper_wordNum_year_classNum_L[:trainPaperNum]
        else:
            trainPaper_wordNum_year_classNum_L = random.sample(trainPaper_wordNum_year_classNum_L, trainPaperNum)
        trainPaper_S = set([i[0] for i in trainPaper_wordNum_year_classNum_L])

        # 计算每类别论文
        class_testPaper_D = {}  # {class:{paperID,..},..}
        for paper, _, _, _ in trainPaper_wordNum_year_classNum_L:
            class_L = self._id_title_author_c1_c2_abstract_year_D[paper][whatClassUsed]
            for c in class_L:
                if c in class_testPaper_D:
                    class_testPaper_D[c].add(paper)
                else:
                    class_testPaper_D[c] = {paper}

        # 筛选满足条件的测试集论文
        testPaper_labelS_wordNum_year_classNum_D = {}  # {paperID:(检索到的论文set,单词数量,年份,类别数量),..}
        alltestPaperClass_S = set()
        alltestPaperClassAll_S = set()  # 包含没有在训练集中出现的分类
        for paperId, paperInfor_L in tqdm(self._id_title_author_c1_c2_abstract_year_D.items(), mName+'-筛选测试集论文'):
            if paperId in trainPaper_S:  # 不能是训练集论文
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
            if not paperTitleWordNum_T[0] <= len(title.split()) <= paperTitleWordNum_T[1]:  # 论文title单词数要求
                continue
            if not paperAbstractWordNum_T[0] <= len(abstract.split()) <= paperAbstractWordNum_T[1]:  # 论文abstract单词数要求
                continue
            labelPaper_S = set()
            testPaperClassL = []
            for c in paperInfor_L[whatClassUsed]:
                if c in class_testPaper_D:
                    labelPaper_S |= class_testPaper_D[c]
                    testPaperClassL.append(c)
            if minRetrievalPaperNum > len(labelPaper_S):  # 满足测试论文最少检索到的论文数量
                continue
            testPaper_labelS_wordNum_year_classNum_D[paperId] = (labelPaper_S, wordNum, year, len(testPaperClassL))
            alltestPaperClass_S |= set(testPaperClassL)
            alltestPaperClassAll_S |= set(paperInfor_L[whatClassUsed])
        testPaperNumAll = len(testPaper_labelS_wordNum_year_classNum_D)  # 所有满足条件的测试集论文数量
        testPaper_labelS_wordNum_year_classNum_L = sorted(testPaper_labelS_wordNum_year_classNum_D.items(), key=lambda t: len(t[1][0]))
        testPaperMaxLabelNumAll = len(testPaper_labelS_wordNum_year_classNum_L[-1][1][0])  # 所有满足条件的测试集论文中标签最多的论文的标签数量
        testPaper_labelS_wordNum_year_classNum_L = testPaper_labelS_wordNum_year_classNum_L[:testPaperNum]

        # 统计信息
        trainPaperYearMaxMinAve_L = [0, 1000000, 0]  # 训练集论文年份
        trainPaperWordNumMaxMinAve_L = [0, 1000000, 0]
        trainClassNumMaxMinAve_L = [0, 1000000, 0]
        classTrainPaperNumMaxMinAve_L = [0, 1000000, 0]  # 测试集所有论文的分类的测试集论文数量

        testPaperYearMaxMinAve_L = [0, 1000000, 0]
        testPaperWordMaxMinAve_L = [0, 1000000, 0]
        testPaperLabelNumMaxMinAve_L = [0, 1000000, 0]
        testPaperClassNumMaxMinAve_L = [0, 1000000, 0]
        # 训练集论文统计信息
        for _, wordNum, year, classNum in tqdm(trainPaper_wordNum_year_classNum_L, mName+'-统计训练集论文信息'):
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
            # 分类数
            if trainClassNumMaxMinAve_L[0] < classNum:
                trainClassNumMaxMinAve_L[0] = classNum
            if trainClassNumMaxMinAve_L[1] > classNum:
                trainClassNumMaxMinAve_L[1] = classNum
            trainClassNumMaxMinAve_L[2] += classNum
        trainPaperYearMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        trainPaperWordNumMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        trainClassNumMaxMinAve_L[2] /= len(trainPaper_wordNum_year_classNum_L)
        # 测试集所有论文的分类的测试集论文数量
        for _, s in class_testPaper_D.items():
            if classTrainPaperNumMaxMinAve_L[0] < len(s):
                classTrainPaperNumMaxMinAve_L[0] = len(s)
            if classTrainPaperNumMaxMinAve_L[1] > len(s):
                classTrainPaperNumMaxMinAve_L[1] = len(s)
            classTrainPaperNumMaxMinAve_L[2] += len(s)
        classTrainPaperNumMaxMinAve_L[2] /= len(class_testPaper_D)
        # 测试集统计信息
        for _, v in tqdm(testPaper_labelS_wordNum_year_classNum_L, mName+'-统计测试集论文信息'):
            labelS, wordNum, year, classNum = v
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
            # 论文检索到的论文数
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

        print('满足条件的理论训练集论文数:%d, 满足条件的理论测试集论文数:%d, 理论测试集论文最大标签数:%d' % (trainPaperNumAll, testPaperNumAll, testPaperMaxLabelNumAll))
        print('训练集论文MaxMinAve年份:%s, 训练集论文MaxMinAve词数:%s, 训练集论文MaxMinAve分类:%s, '
              '训练集论文总数:%d, 训练集分类种数:%d, 分类MaxMinAve训练集论文数:%s' %
              (str(trainPaperYearMaxMinAve_L), str(trainPaperWordNumMaxMinAve_L), str(trainClassNumMaxMinAve_L),
               len(trainPaper_wordNum_year_classNum_L), len(class_testPaper_D), str(classTrainPaperNumMaxMinAve_L)))
        print('测试集论文MaxMinAve年份:%s, 测试集论文MaxMinAve词数:%s, 测试集论文MaxMinAve标准答案:%s, 测试集论文MaxMinAve分类:%s, '
              '测试集论文数:%d, 测试集论文分类种数:%d, 测试集论文分类种数(包含没有在训练集中出现的分类):%d' %
              (str(testPaperYearMaxMinAve_L), str(testPaperWordMaxMinAve_L), str(testPaperLabelNumMaxMinAve_L), str(testPaperClassNumMaxMinAve_L),
               len(testPaper_labelS_wordNum_year_classNum_L), len(alltestPaperClass_S), len(alltestPaperClassAll_S)))

        self._trainPaper_wordNum_year_classNum_L = trainPaper_wordNum_year_classNum_L
        self._testPaper_labelS_wordNum_year_classNum_L = testPaper_labelS_wordNum_year_classNum_L

    def savePaperFolder(self, path, labelName='测试论文编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=False):
        assert self._trainPaper_wordNum_year_classNum_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_classNum_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path), '删除原文件夹内容...'):
                    os.remove(path + '/' + i)
        for paperID, _, _, _ in tqdm(self._trainPaper_wordNum_year_classNum_L, mName+'-写入训练集论文文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(text)
        for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName+'-写入测试集论文文本'):
            with open(path+'/'+paperID+'.text', 'w', encoding='utf-8') as w:
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(text)
        with open(path + '/' + labelName, 'w', encoding='utf-8') as w:
            for paperID, v in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName+'-写入测试集论文标签(标准答案)'):
                labelS = v[0]
                w.write(paperID+'\t'+'\t'.join(labelS)+'\n')

    def savePaperFile(self, address, segTitleAbstract='\t'):
        assert self._trainPaper_wordNum_year_classNum_L, '没有训练集信息可以存储!'
        assert self._testPaper_labelS_wordNum_year_classNum_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        testPaper_label_D = {i: v[0] for i, v in self._testPaper_labelS_wordNum_year_classNum_L}  # {测试集论文:训练集论文答案set,..}
        with open(address, 'w', encoding='utf-8') as w:
            w.write(str(testPaper_label_D)+'\n')
            for paperID, _, _, _ in tqdm(self._trainPaper_wordNum_year_classNum_L, mName + '-写入训练集论文文本'):
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(paperID + '\t' + text + '\n')
            for paperID, _ in tqdm(self._testPaper_labelS_wordNum_year_classNum_L, mName + '-写入测试集论文文本'):
                paperInfor_L = self._id_title_author_c1_c2_abstract_year_D[paperID]
                text = paperInfor_L[0] + segTitleAbstract + paperInfor_L[4]
                w.write(paperID + '\t' + text + '\n')


class PaperAnalysis:
    def __init__(self):
        self._id_title_author_c1_c2_abstract_year_L = None

    def startAnalysis(self,xmlFolderPath:str,savePath = None):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        allPath_L = self._getAllXmlPath(xmlFolderPath)
        allPaper_L = [] # [[编号,标题l,作者l,分类l,摘要l,日期l],..]
        for path in tqdm(allPath_L,mName+'-读取xml文件'):
            with open(path,'r',encoding='utf-8') as r:
                text = r.read()
                allPaper_L += self._analysisXmlText(text)
        # 统计与清洗
        id_title_author_c1_c2_abstract_year_L = []  # [[id,标题,[作者,..],[subject分类,..],[MSC分类,..],摘要,年份],..]  无则为None
        noTitleSum = 0
        noAuthorSum = 0
        noC1Sum = 0
        noC2Sum = 0
        noAbstractSum = 0
        noYearSum = 0
        for paper in tqdm(allPaper_L,mName+'-提取论文信息'):
            id = paper[0]
            # 标题
            title = None
            for i in paper[1]:
                title = self._clean(paper[1][0])
                break
            if not title:
                noTitleSum += 1
            # 作者
            author_L = []
            for i in paper[2]:
                if len(re.findall('[a-zA-Z]', i)) == 0:  # 如果不含字母不是作者
                    continue
                i=i.lower() # 防止形成的文件名不区分大小写而覆盖
                author_L.append(i)
            if not author_L:
                noAuthorSum += 1
            # 分类
            c1_L = [] # subject
            c2_L = [] # MSC class
            for i in paper[3]:
                if len(re.findall('[a-z]',i)) > 0: # 如果含有小写字母
                    if len(re.findall('Primary:|Secondary:|primary:|secondary:',i)) == 0:
                        c1_L.append(i.strip())
                    else:
                        for j in re.findall('(?<=[^0-9A-Za-z.])[0-9A-Z.]+?(?=[^0-9A-Za-z.])',i):
                            c2_L.append(j.strip('.'))
                    continue
                for j in re.split('[,;\s]+',i):
                    c2_L.append(j.strip().strip(' .'))
            if not c1_L:
                noC1Sum += 1
            if not c2_L:
                noC2Sum += 1
            # 摘要
            abstract = None
            for i in paper[4]:
                if re.findall('\*\*\*\s*Comments:|\*\*\*\s*comments:','***'+i):
                    continue
                abstract = self._clean(i)
                break
            if len(paper[4])==1 and not abstract:
                abstract = paper[4][0]
            if not abstract:
                noAbstractSum += 1
            # 年份
            year = None
            if paper[5]:
                year = int(paper[5][-1].split('-')[0]) # -1表示最新的一年
            if not year:
                noYearSum += 1
            id_title_author_c1_c2_abstract_year_L.append([id,title,author_L,c1_L,c2_L,abstract,year])
        print('论文总数:%d, 无标题论文数:%d, 无作者论文数:%d, 无subject分类论文数:%d, 无MSC分类论文数:%d, 无摘要论文数:%d, 无年份论文数:%d'%
              (len(allPaper_L),noTitleSum,noAuthorSum,noC1Sum,noC2Sum,noAbstractSum,noYearSum))
        # 保存
        if savePath:
            with open(savePath,'w',encoding='utf-8') as w:
                for i in tqdm(id_title_author_c1_c2_abstract_year_L,'写入论文信息'):
                    w.write(i[0])
                    for j in i[1:]:
                        w.write('\t'+str(j))
                    w.write('\n')
        self._id_title_author_c1_c2_abstract_year_L = id_title_author_c1_c2_abstract_year_L
        return id_title_author_c1_c2_abstract_year_L

    def _clean(self,text:str):
        text = text.strip().lower() # 全部小写
        text = text.replace('\t',' ')
        text = text.replace('\n',' ')
        text = text.replace('\r',' ')
        return text

    def _getAllXmlPath(self,path:str):
        allPath_L = []
        for fileName in os.listdir(path):
            suffix = os.path.splitext(fileName)[1].lower()
            if suffix == '.xml':
                allPath_L.append(path + '/' + fileName)
        return allPath_L

    def _analysisXmlText(self,text:str):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        allPaper_L = [] # [[编号,标题,作者,分类,摘要,日期],..]
        for t in text.split('</record>')[:-1]:
            t=t.replace('\r','')
            t=t.replace('\n','')
            try:
                id = re.findall('(?<=<record id="oai:arXiv.org:)[^"]+',t)[0]
                id = id.replace('/',';') # 为了能够做文件名,进行替换
            except:
                print(t)
                raise 1
            title_L = re.findall('(?<=<dc:title>).+?(?=</dc:title>)',t)
            creator_L = re.findall('(?<=<dc:creator>).+?(?=</dc:creator>)',t)
            subject_L = re.findall('(?<=<dc:subject>).+?(?=</dc:subject>)',t)
            description_L = re.findall('(?<=<dc:description>).+?(?=</dc:description>)',t)
            date_L = re.findall('(?<=<dc:date>).+?(?=</dc:date>)',t)
            allPaper_L.append([id,title_L,creator_L,subject_L,description_L,date_L])
        return allPaper_L

    def readSaveFile(self,path):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        id_title_author_c1_c2_abstract_year_L = []  # [[id,标题,[作者,..],[subject分类,..],[MSC分类,..],摘要,年份],..]  无则为None
        with open(path,'r',encoding='utf-8') as r:
            for line in tqdm(r,mName+'-读取数据文件'):
                line = line.strip().split('\t')
                if len(line)<3:
                    continue

                id = line[0]
                title = line[1]
                author_L = eval(line[2])
                c1_L = eval(line[3])
                c2_L = eval(line[4])
                abstract = line[5]
                year = line[6]

                if title == 'None': title = None
                if abstract == 'None': abstract = None
                if year == 'None':
                    year = None
                else:
                    year = int(year)

                id_title_author_c1_c2_abstract_year_L.append([id, title, author_L, c1_L, c2_L, abstract, year])
        self._id_title_author_c1_c2_abstract_year_L = id_title_author_c1_c2_abstract_year_L
        return id_title_author_c1_c2_abstract_year_L

    def writePaperText(self,path):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_title_author_c1_c2_abstract_year_L,'没有论文信息!'
        with open(path,'w',encoding='utf-8') as w:
            for id,title,author_L,c1_L,c2_L,abstract,year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出'):
                w.write(id+'\t'+title+'\t'+abstract+'\n')

    def writeCorpus(self,path, textFilter=None):
        '''
        没有过滤器则输出纯粹的预料(双倍), 输出2遍, 第一遍原文, 第二遍将非字母数字替换为空格, 用于词向量训练
        '''
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        assert self._id_title_author_c1_c2_abstract_year_L, '没有论文信息!'
        with open(path, 'w', encoding='utf-8') as w:
            if not textFilter:
                for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出第一遍'):
                    w.write(title + '\n')
                    w.write(abstract + '\n')
                w.write('\n')
                for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出第二遍'):
                    w.write(re.sub('[^0-9a-zA-Z]+',' ',title) + '\n')
                    w.write(re.sub('[^0-9a-zA-Z]+',' ',abstract) + '\n')
            else:
                for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName+'输出'):
                    w.write(textFilter(title) + '\n')
                    w.write(textFilter(abstract) + '\n')

    def write论文_方向_年份_作者_引用论文_被引论文广义表(self,address,whatClassUsed= 'subject'):
        assert self._id_title_author_c1_c2_abstract_year_L, '没有论文信息!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        论文_方向_年份_作者_引用论文_被引论文广义表 = {}
        for id, title, author_L, c1_L, c2_L, abstract, year in tqdm(self._id_title_author_c1_c2_abstract_year_L,mName + '获得广义表'):
            if whatClassUsed == 'subject':
                c = c1_L
            elif whatClassUsed == 'MSC':
                c = c2_L
            else:
                assert False, '分类错误!'
            方向 = {i:None for i in c}
            年份 = {str(year):None} if year else {'0':None}
            作者 = {i:None for i in author_L}
            论文_方向_年份_作者_引用论文_被引论文广义表[id] = [方向,年份,作者,{},{},{},{}]
        二进制流 = pickle.dumps(论文_方向_年份_作者_引用论文_被引论文广义表)
        with open(address, 'wb') as w:
            w.write(二进制流)


if __name__ == '__main__':
    PaperAnalysis_obj = PaperAnalysis()
    # id_title_author_c1_c2_abstract_year_L = PaperAnalysis_obj.startAnalysis(r'K:\数据\arxiv',
    #                             r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_编号_标题_作者_分类1_分类2_摘要_日期.txt')
    id_title_author_c1_c2_abstract_year_L = PaperAnalysis_obj.readSaveFile(
        r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_编号_标题_作者_分类1_分类2_摘要_日期.txt')
    # PaperAnalysis_obj.writePaperText(r'data\arxiv论文编号_英文题目摘要.txt')
    PaperAnalysis_obj.writeCorpus(r'data\all arxiv\ab_arxivCorpus.text', textFilter=句子清洗)
    # PaperAnalysis_obj.write论文_方向_年份_作者_引用论文_被引论文广义表(address=r'D:\data\code\data\1-RAP\arxiv数据\ab_arxiv_论文_方向_年份_作者_引用论文_被引论文广义表.pkl',
    #                                                 whatClassUsed= 'subject')

    # PaperFilter_obj = PaperFilter(id_title_author_c1_c2_abstract_year_L)
    # PaperFilter_obj.filter(trainPaperNum=20000, trainPaperYear_T=(0, 2015), paperTitleWordNum_T=(5, 30),
    #                        testPaperNum=1000, testPaperYear_T=(2016, 2019), paperAbstractWordNum_T=(200, 500),
    #                        whatClassUsed='subject', minRetrievalPaperNum=20, orderChoiceTrainPaper=False)
    # print('按任意键继续保存文件...')
    # os.system("pause")
    # # PaperFilter_obj.savePaperFolder(path='data/IR arxiv/dataset/', labelName='测试论文编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=True)
    # PaperFilter_obj.savePaperFile(address='data/IR arxiv/dataset.text', segTitleAbstract='\t')
