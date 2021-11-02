import re
import html


def 句子清洗(句子: str):
    # 全部小写
    句子 = 句子.lower()
    # 去除html标签
    句子 = re.sub('<[^<>]+>', '', 句子)
    # html字符转义
    句子 = html.unescape(句子)
    # 按符号分割
    句子 = ' ' + ' '.join(re.split(r'[!@#$%^&*()_+\-=\[\]\\{\}|;:"<>?,./~\s]+', 句子)) + ' '
    # 句子 = ' ' + ' '.join(re.split(r'[!;:"?,.\s]+',句子)) + ' '
    # 删除不含字母的词
    句子 = re.sub(' [^a-z]+ ', ' ', 句子)
    # 删除含有奇怪符号的词
    # 句子 = re.sub(' [^ ]+[^a-z0-9\'][^ ]+ ',' ',句子)
    return 句子.strip()


def html字符处理(句子: str):
    # 全部小写
    句子 = 句子.lower()
    # 去除html标签
    句子 = re.sub('<[^<>]+>', ' ', 句子)
    # html字符转义
    句子 = html.unescape(句子)
    # 多空格转单空格
    句子 = re.sub(r'\s+ ', ' ', 句子)
    return 句子.strip()


def 停用词过滤(word_L, stopwords_S):
    word_L2 = []
    if isinstance(stopwords_S, str):
        stopwords2_S = set()
        with open(stopwords_S, 'r', encoding='utf-8') as r:
            for line in r:
                line = line.strip()
                stopwords2_S.add(line)
                stopwords2_S.add(line.lower())
        stopwords_S = stopwords2_S
    if stopwords_S:
        for w in word_L:
            if w not in stopwords_S:
                word_L2.append(w)
    return word_L2, stopwords_S
