import pickle
from tqdm import tqdm
import random
import sys
import os
import re
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 提取关键信息:
    def __init__(self, patentPath=None, uspatentcitationPath=None, claimPath=None, 已提取信息地址=None):
        if 已提取信息地址:
            print('从地址中读取信息...')
            with open(已提取信息地址.encode('utf-8'), 'rb') as r:
                id_year_title_abstract_ref_D = pickle.load(r)
            print('专利数量: %d' % (len(id_year_title_abstract_ref_D)))
        else:
            id_year_title_abstract_ref_D = {}  # {id:[year,title,abstract,{ref,..}],..}, title由标题和摘要构成, abstract由claim第一段构成
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
                    year_title_abstract_ref = [None, None, [], set()]  # claim 列表会去除
                    year_title_abstract_ref[1] = title + '. ' + abstract
                    year_title_abstract_ref[0] = year
                    id_year_title_abstract_ref_D[patent_id] = year_title_abstract_ref
                    有效专利 += 1
            print('提取有效专利:%d, 字典中专利数:%d' % (有效专利, len(id_year_title_abstract_ref_D)))
            # 提取 claim
            有效专利 = 0
            with open(claimPath, 'r', encoding='utf-8') as r:
                r.readline()  # ['uuid', 'patent_id', 'text', 'dependent', 'sequence', 'exemplary']
                for i, line in tqdm(enumerate(r), '提取 claim.tsv'):
                    line = line.strip('\r\n').split('\t')
                    if len(line) != 6:
                        continue
                    patent_id = line[1].strip()
                    if patent_id not in id_year_title_abstract_ref_D:
                        continue
                    claim = line[2].strip().lower()
                    sequence = int(line[4].strip())  # 表示第几段claim
                    exemplary = line[5].strip().lower()  # 是否为示范权力要求之一, true or false
                    # 可以根据dependent和exemplary筛选claim
                    if sequence != 1:
                        continue
                    if not patent_id or not claim:
                        continue
                    id_year_title_abstract_ref_D[patent_id][2].append((sequence, claim))
                    有效专利 += 1
            print('提取有效行数:%d' % 有效专利)
            # 提取 引用
            有效专利 = 0
            with open(uspatentcitationPath, 'r', encoding='utf-8') as r:
                r.readline()  # ['uuid', 'patent_id', 'citation_id', 'date', 'name', 'kind', 'country', 'category', 'sequence']
                for i, line in tqdm(enumerate(r), '提取 uspatentcitation.tsv'):
                    line = line.strip('\r\n').split('\t')
                    if len(line) != 9:
                        continue
                    patent_id = line[1].strip()
                    if patent_id not in id_year_title_abstract_ref_D:
                        continue
                    citation_id = line[2].strip()
                    if citation_id not in id_year_title_abstract_ref_D:
                        continue
                    category = line[7].strip()
                    if category != 'cited by examiner':  # 只保留审查员提供的引用
                        continue
                    id_year_title_abstract_ref_D[patent_id][3].add(citation_id)
                    有效专利 += 1
            print('提取有效引用:%d' % 有效专利)
            # 剔除有空属性的专利
            delete = []
            for i, year_title_abstract_ref in tqdm(id_year_title_abstract_ref_D.items(), '整理专利属性'):
                claim = year_title_abstract_ref[2]
                ref = year_title_abstract_ref[3]
                if not claim or not ref:
                    delete.append(i)
                claim = sorted(claim, key=lambda t: t[0])  # 按句子位置排序
                claim = ' '.join([i for _, i in claim])  # 去掉序号, 并合并
                year_title_abstract_ref[2] = claim
            for i in delete:
                del id_year_title_abstract_ref_D[i]
            print('最终提取到%d篇全属性专利' % len(id_year_title_abstract_ref_D))
        self._id_year_title_abstract_ref_D = id_year_title_abstract_ref_D

    def statistics(self):
        titleLen_max_min_avg = [0, 10**10, 0.]
        abstractLen_max_min_avg = [0, 10**10, 0.]
        patentRefNum_max_min_avg = [0, 10**10, 0.]
        yearPatentNum_D = {}
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

            if len(references) > patentRefNum_max_min_avg[0]:
                patentRefNum_max_min_avg[0] = len(references)
            if len(references) < patentRefNum_max_min_avg[1]:
                patentRefNum_max_min_avg[1] = len(references)
            patentRefNum_max_min_avg[2] += len(references)

            if year in yearPatentNum_D:
                yearPatentNum_D[year] += 1
            else:
                yearPatentNum_D[year] = 1
        titleLen_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        abstractLen_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        patentRefNum_max_min_avg[2] /= len(self._id_year_title_abstract_ref_D)
        print('titleLen_max_min_avg: %s, abstractLen_max_min_avg: %s, patentRefNum_max_min_avg: %s' %
              (str(titleLen_max_min_avg), str(abstractLen_max_min_avg), str(patentRefNum_max_min_avg)))
        print('年份_专利数: %s' % str(sorted(yearPatentNum_D.items(), key=lambda t: int(t[0]), reverse=True)))

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
        assert self._id_year_title_abstract_ref_D, '没有专利信息!'
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


class PatentFilter:  # 这里的词数量统计不使用句子清洗
    def __init__(self, id_year_title_abstract_ref_D):
        self._id_year_title_abstract_ref_D = id_year_title_abstract_ref_D  # {id:[year,title,abstract,{ref,..}],..}
        self._trainPatent_wordNum_year_L = {}  # [(patentID,单词数量,年份),..]
        self._testPatent_labelS_wordNum_year_L = {}  # {patentID:(引文set,单词数量,年份),..}

    def filter(self, trainPatentNum=10000, trainPatentYear_T=(0, 2015), patentTitleWordNum_T=(5, 50),
               testPatentNum=100, testPatentYear_T=(2016, 2019), patentAbstractWordNum_T=(150, 1000),
               minRefPatentNum=20, orderChoiceTrainPatent=True):
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name

        # 筛选满足条件的训练集专利, 保证提出的专利都是满足单词要求的
        trainPatent_wordNum_year_L = []  # [(patentID,单词数量,年份),..]
        for i, year_title_abstract_ref in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'-筛选训练集专利'):
            year = int(year_title_abstract_ref[0])
            title = year_title_abstract_ref[1]
            abstract = year_title_abstract_ref[2]
            text = title + ' ' + abstract
            if not trainPatentYear_T[0] <= year <= trainPatentYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not patentTitleWordNum_T[0] <= len(title.split()) <= patentTitleWordNum_T[1]:  # 专利title单词数要求
                continue
            if not patentAbstractWordNum_T[0] <= len(abstract.split()) <= patentAbstractWordNum_T[1]:  # 专利abstract单词数要求
                continue
            trainPatent_wordNum_year_L.append((i, wordNum, year))
        trainPatentNumAll = len(trainPatent_wordNum_year_L)  # 所有满足条件的训练集的专利数量
        trainPatent_S = set([i[0] for i in trainPatent_wordNum_year_L])  # 用于筛选训练集引文

        # 筛选满足条件的测试集专利, 去除不在以上训练集中的引文
        testPatent_labelS_wordNum_year_D = {}  # {patentID:(引文set,单词数量,年份),..}
        for i, year_title_abstract_ref in tqdm(self._id_year_title_abstract_ref_D.items(), mName+'-筛选测试集专利'):
            if i in trainPatent_S:  # 不能是训练集专利
                continue
            year = int(year_title_abstract_ref[0])
            title = year_title_abstract_ref[1]
            abstract = year_title_abstract_ref[2]
            references = year_title_abstract_ref[3]  # list
            text = title + ' ' + abstract
            if not testPatentYear_T[0] <= year <= testPatentYear_T[1]:  # 时间限制
                continue
            if not title or not abstract:  # 标题和摘要不能缺少
                continue
            wordNum = len(text.split())
            if not patentTitleWordNum_T[0] <= len(title.split()) <= patentTitleWordNum_T[1]:  # 专利title单词数要求
                continue
            if not patentAbstractWordNum_T[0] <= len(abstract.split()) <= patentAbstractWordNum_T[1]:  # 专利abstract单词数要求
                continue
            labelPatent_S = set(references) & trainPatent_S  # 必须在训练集中
            if minRefPatentNum > len(labelPatent_S):  # 满足测试专利最少检索到的专利数量
                continue
            testPatent_labelS_wordNum_year_D[i] = (labelPatent_S, wordNum, year)
        testPatentNumAll = len(testPatent_labelS_wordNum_year_D)  # 所有满足条件的测试集专利数量
        testPatent_labelS_wordNum_year_L = sorted(testPatent_labelS_wordNum_year_D.items(), key=lambda t: len(t[1][0]))
        testPatentMaxLabelNumAll = len(testPatent_labelS_wordNum_year_L[-1][1][0])  # 所有满足条件的测试集专利中标签最多的专利的标签数量
        testPatent_labelS_wordNum_year_L = testPatent_labelS_wordNum_year_L[:testPatentNum]
        allRef_S = set()  # 用于筛选训练集
        for _, (ref_S, _, _) in testPatent_labelS_wordNum_year_L:
            allRef_S |= ref_S
        print('用于groundtruth的训练集文本数量:%d' % len(allRef_S))

        # 获得真正的训练集, 合并引文,然后加入剩下需要的专利
        trainPatent_L = list(allRef_S)
        if orderChoiceTrainPatent:  # 不随机
            i = 0
            while len(trainPatent_L) < trainPatentNum:
                j = trainPatent_wordNum_year_L[i][0]
                if j not in allRef_S:
                    trainPatent_L.append(j)
                i += 1
        else:  # 随机挑选
            if trainPatentNum > len(trainPatent_L):
                剩余专利 = list(trainPatent_S - allRef_S)
                random.shuffle(剩余专利)
                trainPatent_L += 剩余专利[:trainPatentNum-len(trainPatent_L)]
        trainPatent_S = set(trainPatent_L)  # 新的训练集专利id集合
        trainPatent_wordNum_year_L_real = []
        for i, wordNum, year in trainPatent_wordNum_year_L:  # 真正的训练集挑选出来
            if i in trainPatent_S:
                trainPatent_wordNum_year_L_real.append((i, wordNum, year))
        trainPatent_wordNum_year_L = trainPatent_wordNum_year_L_real

        # 统计信息
        trainPatentYearMaxMinAve_L = [0, 1000000, 0]  # 训练集专利年份
        trainPatentWordNumMaxMinAve_L = [0, 1000000, 0]

        testPatentYearMaxMinAve_L = [0, 1000000, 0]
        testPatentWordMaxMinAve_L = [0, 1000000, 0]
        testPatentLabelNumMaxMinAve_L = [0, 1000000, 0]
        # 训练集专利统计信息
        for _, wordNum, year in tqdm(trainPatent_wordNum_year_L, mName+'-统计训练集专利信息'):
            # 专利年份
            if trainPatentYearMaxMinAve_L[0] < year:
                trainPatentYearMaxMinAve_L[0] = year
            if trainPatentYearMaxMinAve_L[1] > year:
                trainPatentYearMaxMinAve_L[1] = year
            trainPatentYearMaxMinAve_L[2] += year
            # 专利词数
            if trainPatentWordNumMaxMinAve_L[0] < wordNum:
                trainPatentWordNumMaxMinAve_L[0] = wordNum
            if trainPatentWordNumMaxMinAve_L[1] > wordNum:
                trainPatentWordNumMaxMinAve_L[1] = wordNum
            trainPatentWordNumMaxMinAve_L[2] += wordNum
        trainPatentYearMaxMinAve_L[2] /= len(trainPatent_wordNum_year_L)
        trainPatentWordNumMaxMinAve_L[2] /= len(trainPatent_wordNum_year_L)
        # 测试集统计信息
        for _, v in tqdm(testPatent_labelS_wordNum_year_L, mName+'-统计测试集专利信息'):
            labelS, wordNum, year = v
            # 专利年份
            if testPatentYearMaxMinAve_L[0] < year:
                testPatentYearMaxMinAve_L[0] = year
            if testPatentYearMaxMinAve_L[1] > year:
                testPatentYearMaxMinAve_L[1] = year
            testPatentYearMaxMinAve_L[2] += year
            # 专利词数
            if testPatentWordMaxMinAve_L[0] < wordNum:
                testPatentWordMaxMinAve_L[0] = wordNum
            if testPatentWordMaxMinAve_L[1] > wordNum:
                testPatentWordMaxMinAve_L[1] = wordNum
            testPatentWordMaxMinAve_L[2] += wordNum
            # 引文数
            if testPatentLabelNumMaxMinAve_L[0] < len(labelS):
                testPatentLabelNumMaxMinAve_L[0] = len(labelS)
            if testPatentLabelNumMaxMinAve_L[1] > len(labelS):
                testPatentLabelNumMaxMinAve_L[1] = len(labelS)
            testPatentLabelNumMaxMinAve_L[2] += len(labelS)
        testPatentYearMaxMinAve_L[2] /= len(testPatent_labelS_wordNum_year_L)
        testPatentWordMaxMinAve_L[2] /= len(testPatent_labelS_wordNum_year_L)
        testPatentLabelNumMaxMinAve_L[2] /= len(testPatent_labelS_wordNum_year_L)

        print('满足条件的理论训练集专利数:%d, 满足条件的理论测试集专利数:%d, 理论测试集专利最大标签数:%d' % (trainPatentNumAll, testPatentNumAll, testPatentMaxLabelNumAll))
        print('训练集专利MaxMinAve年份:%s, 训练集专利MaxMinAve词数:%s, 训练集专利总数:%d' %
              (str(trainPatentYearMaxMinAve_L), str(trainPatentWordNumMaxMinAve_L), len(trainPatent_wordNum_year_L)))
        print('测试集专利MaxMinAve年份:%s, 测试集专利MaxMinAve词数:%s, 测试集专利MaxMinAve引文数:%s, 测试集专利数:%d' %
              (str(testPatentYearMaxMinAve_L), str(testPatentWordMaxMinAve_L), str(testPatentLabelNumMaxMinAve_L), len(testPatent_labelS_wordNum_year_L)))

        self._trainPatent_wordNum_year_L = trainPatent_wordNum_year_L
        self._testPatent_labelS_wordNum_year_L = testPatent_labelS_wordNum_year_L

    def savePatentFolder(self, path, labelName='测试专利编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=False):
        assert self._trainPatent_wordNum_year_L, '没有训练集信息可以存储!'
        assert self._testPatent_labelS_wordNum_year_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            if deletePathFile:
                for i in tqdm(os.listdir(path), '删除原文件夹内容...'):
                    os.remove(path + '/' + i)
        for patentID, _, _ in tqdm(self._trainPatent_wordNum_year_L, mName+'-写入训练集专利文本'):
            with open(path+'/'+patentID+'.text', 'w', encoding='utf-8') as w:
                patentInfor_L = self._id_year_title_abstract_ref_D[patentID]
                text = patentInfor_L[1] + segTitleAbstract + patentInfor_L[2]
                w.write(text)
        for patentID, _ in tqdm(self._testPatent_labelS_wordNum_year_L, mName+'-写入测试集专利文本'):
            with open(path+'/'+patentID+'.text', 'w', encoding='utf-8') as w:
                patentInfor_L = self._id_year_title_abstract_ref_D[patentID]
                text = patentInfor_L[1] + segTitleAbstract + patentInfor_L[2]
                w.write(text)
        with open(path + '/' + labelName, 'w', encoding='utf-8') as w:
            for patentID, v in tqdm(self._testPatent_labelS_wordNum_year_L, mName+'-写入测试集专利标签(标准答案)'):
                labelS = v[0]
                w.write(patentID+'\t'+'\t'.join(labelS)+'\n')

    def savePatentFile(self, address, segTitleAbstract='\t'):
        assert self._trainPatent_wordNum_year_L, '没有训练集信息可以存储!'
        assert self._testPatent_labelS_wordNum_year_L, '没有测试集信息可以存储!'
        mName = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        testPatent_label_D = {i: v[0] for i, v in self._testPatent_labelS_wordNum_year_L}  # {测试集专利:训练集专利答案set,..}
        with open(address, 'w', encoding='utf-8') as w:
            w.write(str(testPatent_label_D)+'\n')  # 出现utf-8写入错误可以考虑多选取几次,选到没有错误的文本
            for patentID, _, _ in tqdm(self._trainPatent_wordNum_year_L, mName + '-写入训练集专利文本'):
                patentInfor_L = self._id_year_title_abstract_ref_D[patentID]
                text = patentInfor_L[1] + segTitleAbstract + patentInfor_L[2]
                w.write(patentID + '\t' + text + '\n')
            for patentID, _ in tqdm(self._testPatent_labelS_wordNum_year_L, mName + '-写入测试集专利文本'):
                patentInfor_L = self._id_year_title_abstract_ref_D[patentID]
                text = patentInfor_L[1] + segTitleAbstract + patentInfor_L[2]
                w.write(patentID + '\t' + text + '\n')


if __name__ == '__main__':
    # 提取关键信息_obj = 提取关键信息(patentPath=r'F:\data\USPTO\patent.tsv',
    #                     uspatentcitationPath=r'F:\data\USPTO\uspatentcitation.tsv',
    #                     claimPath=r'F:\data\USPTO\claim.tsv')
    # 提取关键信息_obj.saveFile(r'data\all USPTO\bb_id_year_title_abstract_ref_D.pkl')
    提取关键信息_obj = 提取关键信息(已提取信息地址=r'data/all USPTO/bb_id_year_title_abstract_ref_D.pkl')
    # 提取关键信息_obj.statistics()

    # 开始过滤
    PatentFilter_obj = PatentFilter(提取关键信息_obj.id_year_title_abstract_ref_D)
    PatentFilter_obj.filter(trainPatentNum=20000, trainPatentYear_T=(0, 2016), patentTitleWordNum_T=(100, 300),
                           testPatentNum=1000, testPatentYear_T=(2017, 3000), patentAbstractWordNum_T=(150, 400),
                           minRefPatentNum=8, orderChoiceTrainPatent=False)
    print('按任意键继续保存文件...')
    os.system("pause")  # linux 不暂停
    # PatentFilter_obj.savePatentFolder(path='data/CR USPTO/dataset/', labelName='测试专利编号_训练集编号.text', segTitleAbstract='\t', deletePathFile=True)
    PatentFilter_obj.savePatentFile(address='data/CR USPTO/dataset.text', segTitleAbstract='\t')
