import os


def word2vec():
    os.system(
        'time /home/tansc/c/word2vec/word2vec '
        '-train "/home/tansc/python/paper/text-representation/data/TC AGnews/aj_dataset corpus.text" '
        '-output "/home/tansc/python/paper/text-representation/data/TC AGnews/ak_Corpus_vectors.text" '
        '-cbow 0 -size 100 -window 10 -negative 5 -hs 1 '
        '-sample 1e-4 -threads 62 -binary 0 -iter 100 -min-count 1  '
        '-save-vocab "/home/tansc/python/paper/text-representation/data/TC AGnews/ak_Corpus_word.text"'
    )


def glove():
    CORPUS = '"/home/tansc/python/paper/text-representation/data/TC AGnews/aj_dataset corpus.text"'
    VOCAB_FILE = '"/home/tansc/python/paper/text-representation/data/TC AGnews/ak_glove-Corpus_word.text"'
    COOCCURRENCE_FILE = '"cooccurrence.bin"'
    COOCCURRENCE_SHUF_FILE = '"cooccurrence.shuf.bin"'
    BUILDDIR = '"/home/tansc/c/GloVe-master/build"'
    SAVE_FILE = '"/home/tansc/python/paper/text-representation/data/TC AGnews/ak_glove-Corpus_vectors.text"'  # 结束后修改.txt后缀
    VERBOSE = '2'
    MEMORY = '32.0'  # 占用内存的软限制，单位GB
    VOCAB_MIN_COUNT = '1'
    VECTOR_SIZE = '100'
    MAX_ITER = '100'
    WINDOW_SIZE = '10'
    BINARY = '0'  # Save output in binary format (0: text, 1: binary, 2: both); default 0
    NUM_THREADS = '62'
    X_MAX = '10'

    print('.'*30 + '1')
    os.system(BUILDDIR + '/vocab_count -min-count ' + VOCAB_MIN_COUNT + ' -verbose ' + VERBOSE + ' < ' + CORPUS + ' > ' + VOCAB_FILE)
    print('.'*30 + '2')
    os.system(BUILDDIR + '/cooccur -memory ' + MEMORY + ' -vocab-file ' + VOCAB_FILE + ' -verbose ' + VERBOSE + ' -window-size ' + WINDOW_SIZE + ' < ' + CORPUS + ' > ' + COOCCURRENCE_FILE)
    print('.' * 30 + '3')
    os.system(BUILDDIR + '/shuffle -memory ' + MEMORY + ' -verbose ' + VERBOSE + ' < ' + COOCCURRENCE_FILE + ' > ' + COOCCURRENCE_SHUF_FILE)
    print('.' * 30 + '4')
    os.system(
        BUILDDIR + '/glove -save-file ' + SAVE_FILE +
        ' -threads ' + NUM_THREADS +
        ' -input-file ' + COOCCURRENCE_SHUF_FILE +
        ' -x-max ' + X_MAX +
        ' -iter ' + MAX_ITER +
        ' -vector-size ' + VECTOR_SIZE +
        ' -binary ' + BINARY +
        ' -vocab-file ' + VOCAB_FILE +
        ' -verbose ' + VERBOSE +
        ' -write-header 1'  # 0=no, 1=yes; writes vocab_size/vector_size as first line
    )


def doc2vecc():
    train = '/home/tansc/python/paper/text-representation/data/CR dblp/ak_doc2vecc-shuf.text'
    test = '/home/tansc/python/paper/text-representation/data/CR dblp/aj_dataset corpus.text'

    # print('.' * 30 + '编译')
    # os.system('gcc /home/tansc/c/doc2vecc/doc2vecc.c -o /home/tansc/c/doc2vecc/doc2vecc -lm -pthread -O3 -march=native -funroll-loops')
    print('.' * 30 + '构建train文件')
    os.system('shuf "%s" > "%s"' % (test, train))
    print('.' * 30 + '开始训练')
    os.system(
        'time /home/tansc/c/doc2vecc/doc2vecc '
        '-train "%s" ' % train +
        '-word "/home/tansc/python/paper/text-representation/data/CR dblp/ak_doc2vecc_wvs.text" '
        '-output "/home/tansc/python/paper/text-representation/data/CR dblp/ak_doc2vecc_dvs.text" '
        '-cbow 0 -size 100 -window 10 -negative 5 -hs 1 '
        '-sample 1e-4 -threads 30 -binary 0 -iter 100 -min-count 1 '
        '-test "%s" ' % test +
        '-sentence-sample 0.1 '
        '-save-vocab "/home/tansc/python/paper/text-representation/data/CR dblp/ak_doc2vecc_words.text"'
    )


if __name__ == '__main__':
    # word2vec()
    # glove()
    doc2vecc()

'''
词向量一般修改6处,结束后还要删除.txt后缀
'''
