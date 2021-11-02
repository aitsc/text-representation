# -*- coding: utf-8 -*-
'''
Skip-thought vectors
python2
修改版(增加bitch的进度显示和中文编码+pkl3读取的支持)
'''
import os

import theano  # ==7.0
import theano.tensor as tensor

import cPickle as pkl
import numpy  # ==1.15.4
import copy
import nltk  # ==3.*

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize

# nltk.download('punkt')  # pip install nltk 后第一次需要执行
from tqdm import tqdm
import pickle
import datetime
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

profile = False

# -----------------------------------------------------------------------------#
# Specify model and table locations here
# -----------------------------------------------------------------------------#
path_to_models = '/u/rkiros/public_html/models/'
path_to_tables = '/u/rkiros/public_html/models/'
# -----------------------------------------------------------------------------#

path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'


def load_model():
    """
    Load the model with saved tables
    """
    # Load model options
    print 'Loading model parameters...'
    with open('%s.pkl' % path_to_umodel, 'rb') as f:
        uoptions = pkl.load(f)
    with open('%s.pkl' % path_to_bmodel, 'rb') as f:
        boptions = pkl.load(f)

    # Load parameters
    uparams = init_params(uoptions)
    uparams = load_params(path_to_umodel, uparams)
    utparams = init_tparams(uparams)
    bparams = init_params_bi(boptions)
    bparams = load_params(path_to_bmodel, bparams)
    btparams = init_tparams(bparams)

    # Extractor functions
    print 'Compiling encoders...'
    embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)
    f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
    embedding, x_mask, ctxw2v = build_encoder_bi(btparams, boptions)
    f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')

    # Tables
    print 'Loading tables...'
    utable, btable = load_tables()

    # Store everything we need in a dictionary
    print 'Packing up...'
    model = {}
    model['uoptions'] = uoptions
    model['boptions'] = boptions
    model['utable'] = utable
    model['btable'] = btable
    model['f_w2v'] = f_w2v
    model['f_w2v2'] = f_w2v2

    return model


def load_tables():
    """
    Load the tables
    """
    words = []
    utable = numpy.load(path_to_tables + 'utable.npy')
    btable = numpy.load(path_to_tables + 'btable.npy')
    f = open(path_to_tables + 'dictionary.txt', 'rb')
    for line in f:
        words.append(line.decode('utf-8').strip())
    f.close()
    utable = OrderedDict(zip(words, utable))
    btable = OrderedDict(zip(words, btable))
    return utable, btable


class Encoder(object):
    """
    Sentence encoder.
    """

    def __init__(self, model):
        self._model = model

    def encode(self, X, use_norm=True, verbose=True, batch_size=128, use_eos=False, use_tqdm=False):
        """
        Encode sentences in the list X. Each entry will return a vector
        """
        return encode(self._model, X, use_norm, verbose, batch_size, use_eos, use_tqdm)


def encode(model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False, use_tqdm=False):
    """
    Encode sentences in the list X. Each entry will return a vector
    """
    # first, do preprocessing
    X = preprocess(X)

    # word dictionary and init
    d = defaultdict(lambda: 0)
    for w in model['utable'].keys():
        d[w] = 1
    ufeatures = numpy.zeros((len(X), model['uoptions']['dim']), dtype='float32')
    bfeatures = numpy.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')

    # length dictionary
    ds = defaultdict(list)
    captions = [s.split() for s in X]
    for i, s in enumerate(captions):
        ds[len(s)].append(i)

    # Get features. This encodes by length, in order to avoid wasting computation
    total = 0
    initial = 0
    for k in ds.keys():
        total += len(ds[k]) / batch_size + 1
    for i, k in enumerate(ds.keys()):
        if verbose:
            print k
        numbatches = len(ds[k]) / batch_size + 1
        if use_tqdm:
            x = tqdm(range(numbatches), 'bitch', initial=initial, total=total, leave=False)
        else:
            x = range(numbatches)
        for minibatch in x:
            caps = ds[k][minibatch::numbatches]

            if use_eos:
                uembedding = numpy.zeros((k + 1, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = numpy.zeros((k + 1, len(caps), model['boptions']['dim_word']), dtype='float32')
            else:
                uembedding = numpy.zeros((k, len(caps), model['uoptions']['dim_word']), dtype='float32')
                bembedding = numpy.zeros((k, len(caps), model['boptions']['dim_word']), dtype='float32')
            for ind, c in enumerate(caps):
                caption = captions[c]
                for j in range(len(caption)):
                    if d[caption[j]] > 0:
                        uembedding[j, ind] = model['utable'][caption[j]]
                        bembedding[j, ind] = model['btable'][caption[j]]
                    else:
                        uembedding[j, ind] = model['utable']['UNK']
                        bembedding[j, ind] = model['btable']['UNK']
                if use_eos:
                    uembedding[-1, ind] = model['utable']['<eos>']
                    bembedding[-1, ind] = model['btable']['<eos>']
            if use_eos:
                uff = model['f_w2v'](uembedding, numpy.ones((len(caption) + 1, len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption) + 1, len(caps)), dtype='float32'))
            else:
                uff = model['f_w2v'](uembedding, numpy.ones((len(caption), len(caps)), dtype='float32'))
                bff = model['f_w2v2'](bembedding, numpy.ones((len(caption), len(caps)), dtype='float32'))
            if use_norm:
                for j in range(len(uff)):
                    uff[j] /= norm(uff[j])
                    bff[j] /= norm(bff[j])
            for ind, c in enumerate(caps):
                ufeatures[c] = uff[ind]
                bfeatures[c] = bff[ind]
        initial += numbatches

    features = numpy.c_[ufeatures, bfeatures]
    return features


def preprocess(text):
    """
    Preprocess text for encoder
    """
    X = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for t in text:
        sents = sent_detector.tokenize(t)
        result = ''
        for s in sents:
            tokens = word_tokenize(s)
            result += ' ' + ' '.join(tokens)
        X.append(result)
    return X


def nn(model, text, vectors, query, k=5):
    """
    Return the nearest neighbour sentences to query
    text: list of sentences
    vectors: the corresponding representations for text
    query: a string to search
    """
    qf = encode(model, [query])
    qf /= norm(qf)
    scores = numpy.dot(qf, vectors.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [text[a] for a in sorted_args[:k]]
    print 'QUERY: ' + query
    print 'NEAREST: '
    for i, s in enumerate(sentences):
        print s, sorted_args[i]


def word_features(table):
    """
    Extract word features into a normalized matrix
    """
    features = numpy.zeros((len(table), 620), dtype='float32')
    keys = table.keys()
    for i in range(len(table)):
        f = table[keys[i]]
        features[i] = f / norm(f)
    return features


def nn_words(table, wordvecs, query, k=10):
    """
    Get the nearest neighbour words
    """
    keys = table.keys()
    qf = table[query]
    scores = numpy.dot(qf, wordvecs.T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    words = [keys[a] for a in sorted_args[:k]]
    print 'QUERY: ' + query
    print 'NEAREST: '
    for i, w in enumerate(words):
        print w


def _p(pp, name):
    """
    make prefix-appended name
    """
    return '%s_%s' % (pp, name)


def init_tparams(params):
    """
    initialize Theano shared variables according to the initial parameters
    """
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def load_params(path, params):
    """
    load parameters
    """
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'gru': ('param_init_gru', 'gru_layer')}


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def init_params(options):
    """
    initialize all parameters needed for the encoder
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])
    return params


def init_params_bi(options):
    """
    initialize all paramters needed for bidirectional encoder
    """
    params = OrderedDict()

    # embedding
    params['Wemb'] = norm_weight(options['n_words_src'], options['dim_word'])

    # encoder: GRU
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_r',
                                              nin=options['dim_word'], dim=options['dim'])
    return params


def build_encoder(tparams, options):
    """
    build an encoder, given pre-computed word embeddings
    """
    # word embedding (source)
    embedding = tensor.tensor3('embedding', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, embedding, options,
                                            prefix='encoder',
                                            mask=x_mask)
    ctx = proj[0][-1]

    return embedding, x_mask, ctx


def build_encoder_bi(tparams, options):
    """
    build bidirectional encoder, given pre-computed word embeddings
    """
    # word embedding (source)
    embedding = tensor.tensor3('embedding', dtype='float32')
    embeddingr = embedding[::-1]
    x_mask = tensor.matrix('x_mask', dtype='float32')
    xr_mask = x_mask[::-1]

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, embedding, options,
                                            prefix='encoder',
                                            mask=x_mask)
    projr = get_layer(options['encoder'])[1](tparams, embeddingr, options,
                                             prefix='encoder_r',
                                             mask=xr_mask)

    ctx = tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)

    return embedding, x_mask, ctx


# some utilities
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


def norm_weight(nin, nout=None, scale=0.1, ortho=True):
    if nout == None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W.astype('float32')


def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    parameter init for GRU
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None, **kwargs):
    """
    Forward pass through GRU layer
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[tensor.alloc(0., n_samples, dim)],
                                non_sequences=[tparams[_p(prefix, 'U')],
                                               tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


if __name__ == '__main__':
    ap = 'data/'
    # dataset = 'IR arxiv/'
    dataset = 'CR dblp/'
    id_ft8bt_D_path = ap + dataset + 'bf_id_ft8bt_D.pkl'  # 输入分割好的句子

    use_sentEm = False  # 表示对文档的每个句子嵌入
    id_ftV8btV_D_path = ap + dataset + 'bg_id_ftV8btV_D.pkl'  # 输出每个句子的嵌入

    use_docEm = True  # 表示将文档所有句子放在一起嵌入，将文档作为句子
    dataset_path = ap + dataset + 'dataset.text'  # 输入数据集, 保证id顺序与数据集一致
    doc_vectors_path = ap + dataset + 'bg_doc_vectors.text'  # 输出文档嵌入

    path_to_models = ap + '-skip-thoughts-model/'
    path_to_tables = ap + '-skip-thoughts-model/'
    path_to_umodel = path_to_models + 'uni_skip.npz'
    path_to_bmodel = path_to_models + 'bi_skip.npz'
    encoder = Encoder(load_model())

    # 获取所有句子
    with open(id_ft8bt_D_path.encode('utf-8'), 'rb') as r:
        print('读取 '+ap+dataset+'id_ft8bt_D ...')
        id_ft8bt_D = pickle.load(r)  # {编号:[[前部句,..],[后部句,..]],..}

    if use_sentEm:
        print('use_sentEm...')
        # 所有句子放在一起
        all_sent_L = []
        id_rangeFB_L = []
        sentV_num = 0
        for ID, (ft_L, bt_L) in id_ft8bt_D.items():
            all_sent_L += ft_L
            all_sent_L += bt_L
            id_rangeFB_L.append([ID, sentV_num, sentV_num + len(ft_L), sentV_num + len(ft_L) + len(bt_L)])
            sentV_num += len(ft_L) + len(bt_L)
        # 所有句子开始计算向量
        print(datetime.datetime.now())
        all_sentV_L = list(encoder.encode(all_sent_L, verbose=False, use_eos=True, use_tqdm=True))
        # 将句子回归到文本编号处
        id_ftV8btV_D = {}  # {编号:[[前部句向量,..],[后部句向量,..]],..}
        for ID, p1, p2, p3 in tqdm(id_rangeFB_L, '构建结果'):
            id_ftV8btV_D[ID] = [all_sentV_L[p1: p2], all_sentV_L[p2: p3]]
        # 保存句嵌入
        info = pickle.dumps(id_ftV8btV_D)
        with open(id_ftV8btV_D_path.encode('utf-8'), 'wb') as w:
            for i in tqdm(range(0, len(info), 10**6), '保存句嵌入'):
                w.write(info[i:i + 10**6])

    if use_docEm:
        print('use_docEm...')
        # 读取id顺序
        textID_L = []
        with open(dataset_path.encode('utf-8'), 'r') as r:
            for i, line in tqdm(enumerate(r), '获取数据集信息'):
                if i == 0:  # 第一行标签
                    continue
                line = line.strip().split('\t')
                textID_L.append(line[0])
        # 句子整合
        text_L = []
        for ID in textID_L:
            ft_L, bt_L = id_ft8bt_D[ID]
            text_L.append(' '.join(ft_L) + ' ' + ' '.join(bt_L))
        # 开始计算向量
        print(datetime.datetime.now())
        doc_vectors_L = list(encoder.encode(text_L, verbose=False, use_eos=True, use_tqdm=True))
        # 保存文档嵌入
        with open(doc_vectors_path.encode('utf-8'), 'w') as w:
            for v in tqdm(doc_vectors_L, '保存文档嵌入'):
                w.write(' '.join([str(i) for i in v]) + '\n')

    print(datetime.datetime.now())
