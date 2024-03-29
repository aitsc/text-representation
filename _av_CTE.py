from _bc_deep_methods import DeepMethods
from _ad_evaluate import IR评估, TC评估
import os
import h5py
from tqdm import tqdm
import tensorflow as tf  # 因为 eval 必须 as tf
from tensorflow.contrib import rnn  # windows=1.11, linux=1.12, cuda=9.0, cuDNN=7.4.2
import random
import sys
import heapq
import pickle
import time
import numpy as np
from multiprocessing import Pool
import math
import importlib
from pprint import pformat, pprint
import logging
from tanshicheng import Draw

logging.basicConfig(level=logging.ERROR)  # 阻止输出初始化句子过长的警告
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 参数文件:
    def __init__(self, 参数地址: str, 参数d: dict):
        print('构建初始参数文件:"%s"' % 参数地址)
        pprint(参数d)
        with open(参数地址, 'w', encoding='utf-8') as w:
            w.write(pformat(参数d))
        self._参数地址 = 参数地址

    @property
    def para(self):
        with open(self._参数地址, 'r', encoding='utf-8') as r:
            参数d = eval(''.join(r.readlines()))
        return 参数d


class 耦合嵌入_tf模型:
    def __init__(self, 模型参数, 初始词向量l=None, 可视化地址=None, 取消可视化=False):
        '''
        词数上限自动加1,从1开始,0是填充
        :param 模型参数d: dict or str 为str表示模型地址,也默认为可视化地址
        :param 初始词向量l: None or [[词,[向量]],..] 为空则随机初始化词向量
        :param 训练可视化地址: str or None
        '''
        assert 模型参数, '缺少模型参数, 是dict或要读取的模型地址.'
        self._可视化w = None
        self._保存模型地址_saver表 = {}
        if isinstance(模型参数, dict):
            assert 'haveTrainingSteps' not in 模型参数, '参数不能包含"haveTrainingSteps"!'
            g, init, self._词_序号d = self._buildingGraph(模型参数, 初始词向量l)
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数['显存占用比']
            tf_config.gpu_options.allow_growth = True  # GPU按需分配，自适应
            self._sess = tf.Session(graph=g, config=tf_config)
            self._sess.run(init)
            if 可视化地址 and not 取消可视化:
                self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
            self._模型参数d = 模型参数.copy()
            self._模型参数d['haveTrainingSteps'] = 0
            pprint(self._模型参数d)
        else:
            self._sess, self._模型参数d, self._词_序号d = self._readModel(模型参数)
            if not 取消可视化:
                if not 可视化地址:
                    可视化地址 = 模型参数
                try:
                    self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
                except:
                    print('模型不含可视化!')

    def _readModel(self, 模型地址):
        '''
        :param 模型地址:
        :return:
        '''
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...', end='')
        sys.stdout.flush()
        startTime = time.time()
        with open((模型地址 + '.parms').encode('utf-8'), 'r', encoding='utf-8') as r:
            模型参数d = eval(r.read())
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数d['显存占用比']
        tf_config.gpu_options.allow_growth = True  # GPU按需分配
        sess = tf.Session(config=tf_config)
        saver = tf.train.import_meta_graph(模型地址 + '.meta')
        saver.restore(sess, 模型地址)
        # 读取词表
        with open((模型地址 + '.w_num_map').encode('utf-8'), 'rb') as r:
            词_序号d = pickle.load(r)
        print('%.2fm' % ((time.time() - startTime) / 60))
        pprint(模型参数d)
        return sess, 模型参数d, 词_序号d

    def _buildingGraph(self, 模型参数d, 初始词向量l):
        '''
        :param 模型参数d: dict 含参数: embedding_dim, title_maxlen, abstract_maxlen, margin, learning_rate, 以及下传model
        :param 初始词向量l: None or [[词,[向量]],..] 为空则随机初始化词向量
        :return: tf.Graph, tf.global_variables_initializer, {词:对应序号,..}
        '''
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...', end='')
        sys.stdout.flush()
        startTime = time.time()
        词_序号d = {'': 0}
        序号_词d = {0: ''}
        词向量矩阵l = [[0.] * 模型参数d['embedding_dim']]
        graph = tf.Graph()
        with graph.as_default():
            if 模型参数d['使用词向量']:
                # 初始化词向量
                词数上限 = 模型参数d['词数上限']
                assert 词数上限, '需要给定词数上限!'
                词数上限 += 1  # 从1开始, 0是填充
                if not 初始词向量l:
                    初始词向量l = []
                if len(初始词向量l) > 词数上限:
                    词数上限 = len(初始词向量l) + 1  # 从1开始, 0是填充向量
                for i, (词, 向量) in enumerate(初始词向量l):
                    i += 1  # 从1开始, 0是填充
                    词_序号d[词] = i
                    序号_词d[i] = 词
                    词向量矩阵l.append(向量)
                for i in range(len(词向量矩阵l), 词数上限):
                    if 模型参数d['词向量固定值初始化'] and -1 <= 模型参数d['词向量固定值初始化'] <= 1:
                        词向量矩阵l.append([模型参数d['词向量固定值初始化']] * 模型参数d['embedding_dim'])
                    else:
                        词向量矩阵l.append([random.uniform(-1, 1) for i in range(模型参数d['embedding_dim'])])
                if 模型参数d['固定词向量']:
                    w_embed = tf.constant(词向量矩阵l, name='w_embed')
                else:
                    w_embed = tf.get_variable('w_embed', initializer=词向量矩阵l)
                tf.constant(词数上限, name='word_count_limit')
                # 词向量信息
                with tf.variable_scope('w_embed_summary'):
                    w_embed_均值 = tf.reduce_mean(w_embed)
                    w_embed_方差 = tf.reduce_mean(tf.square(w_embed - w_embed_均值))
                    w_embed_标准差 = tf.sqrt(w_embed_方差)
                    tf.summary.histogram('w_embed_weights', w_embed)
                    tf.summary.scalar('w_embed_weights_E', w_embed_均值)
                    tf.summary.scalar('w_embed_weights_S', w_embed_标准差)
                # 正例-负例 变量
                title_p = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']], name='title_p')
                abstract_p = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']], name='abstract_p')
                title_n = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']], name='title_n')
                abstract_n = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']], name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)
                # 提供词向量
                title_p_n = tf.nn.embedding_lookup(w_embed, title_p_n)
                abstract_p_n = tf.nn.embedding_lookup(w_embed, abstract_p_n)
            else:
                # 正例-负例 变量
                title_p = tf.placeholder(tf.float32, [None, 模型参数d['title_maxlen'], 模型参数d['句矩阵词dim']], name='title_p')
                abstract_p = tf.placeholder(tf.float32, [None, 模型参数d['abstract_maxlen'], 模型参数d['句矩阵词dim']],
                                            name='abstract_p')
                title_n = tf.placeholder(tf.float32, [None, 模型参数d['title_maxlen'], 模型参数d['句矩阵词dim']], name='title_n')
                abstract_n = tf.placeholder(tf.float32, [None, 模型参数d['abstract_maxlen'], 模型参数d['句矩阵词dim']],
                                            name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)

            # 文本实际长度 变量
            title_len_p = tf.placeholder(tf.int32, [None], name='title_len_p')
            abstract_len_p = tf.placeholder(tf.int32, [None], name='abstract_len_p')
            title_len_n = tf.placeholder(tf.int32, [None], name='title_len_n')
            abstract_len_n = tf.placeholder(tf.int32, [None], name='abstract_len_n')
            title_len = tf.concat([title_len_p, title_len_n], 0)
            abstract_len = tf.concat([abstract_len_p, abstract_len_n], 0)

            # 词向量再处理
            if 模型参数d['词向量tanh']:
                title_p_n = tf.tanh(title_p_n)
                abstract_p_n = tf.tanh(abstract_p_n)
            if 模型参数d['词向量微调']:
                title_p_n = self._词向量微调层(title_p_n)['outputs']
                输出 = self._词向量微调层(abstract_p_n)
                abstract_p_n = 输出['outputs']
                tf.summary.histogram('embedding_weights', 输出['weights'])
                tf.summary.histogram('embedding_biases', 输出['biases'])

            # 使用模型
            isTrain = tf.placeholder(tf.bool, name='isTrain')
            if 模型参数d['LSTM_CNN']['enable']:
                outputs_t, outputs_a = self._m_LSTM_CNN(模型参数d['LSTM_CNN'], title_p_n, abstract_p_n, title_len,
                                                        abstract_len, isTrain)
            else:
                print('至少有一个模型!')
                raise 1
            tf.identity(outputs_t, name="frontText_vec")
            tf.identity(outputs_a, name="backText_vec")

            # 计算距离, 相似度可能为负数
            with tf.variable_scope('sim_f'):
                # GESD, Geometric mean of Euclidean and Sigmoid Dot product
                # sim_p_n = 1 / ((1 + tf.abs(outputs_t - outputs_a)) * (1 + tf.exp(-1-tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1))))
                # AESD, Arithmetic mean of Euclidean and Sigmoid Dot product
                # sim_p_n = 0.5 / (1 + tf.abs(outputs_t - outputs_a)) + 0.5 / (1 + tf.exp(-1-tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1)))
                # 余弦距离, 变这个需要变 标题摘要相似度计算
                outputs_t = tf.nn.l2_normalize(outputs_t, 1, name='title_l2vec')
                outputs_a = tf.nn.l2_normalize(outputs_a, 1, name='abstract_l2vec')
                sim_p_n = tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1, name='sim_p_n')

            # 相似度相关参数
            with tf.variable_scope('sim_result'):
                sim_p, sim_n = tf.split(sim_p_n, num_or_size_splits=2, axis=0)
                正例相似度均值 = tf.reduce_mean(sim_p)
                负例相似度均值 = tf.reduce_mean(sim_n)
                正例相似度方差 = tf.reduce_mean(tf.square(sim_p - 正例相似度均值))
                负例相似度方差 = tf.reduce_mean(tf.square(sim_n - 负例相似度均值))
                正例相似度标准差 = tf.sqrt(正例相似度方差)
                负例相似度标准差 = tf.sqrt(负例相似度方差)
                tf.summary.scalar('E_positive', 正例相似度均值)
                tf.summary.scalar('E_negative', 负例相似度均值)
                tf.summary.scalar('S_positive', 正例相似度标准差)
                tf.summary.scalar('S_negative', 负例相似度标准差)

            # 损失函数
            with tf.variable_scope('loss_f'):
                负正差 = sim_n - sim_p + 模型参数d['margin']
                # 负正差 = tf.maximum(0., sim_n - sim_p + 模型参数d['margin'])
                # 对数负正差 = tf.maximum(0., tf.exp(负正差)-1)
                # 一正差 = (1. - sim_p) * 模型参数d['margin']
                loss_op = tf.reduce_sum(tf.maximum(0.0, 负正差), name='loss_op')
                # loss_op = tf.reduce_sum(对数负正差,name='loss_op')
                # loss_op = tf.reduce_sum(tf.exp(负正差 + 一正差)-1, name='loss_op')
                # loss_op = tf.reduce_sum(tf.maximum(0., tf.exp(负正差 + 一正差)-1), name='loss_op')
                tf.summary.scalar('loss', loss_op)
                tf.summary.scalar('loss_max', tf.reduce_max(负正差))
                对错二分_loss = tf.to_float(tf.greater(0., 负正差), name='right_error_list')
            # 准确率
            with tf.variable_scope('acc_f'):
                对错二分 = tf.to_float(tf.greater(sim_p, sim_n), name='right_error_list')
                accuracy_op = tf.reduce_mean(对错二分, name='accuracy_op')
                tf.summary.scalar('accuracy', accuracy_op)

            # 指标图
            with tf.variable_scope('index'):
                tf.summary.histogram('P_sub_N_sim', sim_p - sim_n)

            # 梯度下降
            global_step = tf.Variable(0)
            if 0 < 模型参数d['AdamOptimizer'] < 1:
                optimizer = tf.train.AdamOptimizer(learning_rate=模型参数d['AdamOptimizer'])
            else:
                学习率最小值 = 模型参数d['learning_rate'] / 模型参数d['学习率最小值倍数']
                learning_rate = tf.maximum(学习率最小值, tf.train.exponential_decay(
                    learning_rate=模型参数d['learning_rate'],
                    global_step=global_step,
                    decay_steps=模型参数d['学习率衰减步数'],
                    decay_rate=模型参数d['学习率衰减率'],
                ))
                tf.summary.scalar('learning_rate', learning_rate)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            # train_op = optimizer.minimize(loss_op, global_step=global_step, name='train_op')
            grads_and_vars = optimizer.compute_gradients(loss_op)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name='train_op')
            # 可视化梯度
            with tf.variable_scope('v_grad'):
                for g, v in grads_and_vars:
                    if g is not None:
                        tf.summary.histogram("{}".format(v.name), g)
                        # tf.summary.scalar("{}".format(v.name), tf.nn.zero_fraction(g))

            # 初始化变量
            init = tf.global_variables_initializer()
            tf.summary.merge_all(name='merged_op')

            # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            # for tensor_name in tensor_name_list: print(tensor_name)

        print('%.2fm' % ((time.time() - startTime) / 60))
        return graph, init, 词_序号d

    def _m_LSTM_CNN(self, 模型参数d, title_p_n, abstract_p_n, title_len, abstract_len, isTrain):
        def biLSTM(x, sequence_length, 模型参数d, keep_prob=1., 可视化=True):
            '''
            描述: 双向LSTM+层叠连接+池化
            :param x: [batch_size, max_time, input_size]
            :param sequence_length: [batch_size]
            :param 模型参数d: dict, 含参数: 各隐藏层数l
            :param keep_prob: float
            :param 可视化: bool
            :return: {outputs:[batch_size, layer_output],cell,..}
            '''
            各隐藏层数l = 模型参数d['biLSTM_各隐层数']
            outputs = x
            biLSTM池化方法 = eval(模型参数d['biLSTM池化方法'])
            输出 = {}
            for i, 隐层数 in enumerate(各隐藏层数l):
                input_keep_prob = keep_prob
                output_keep_prob = keep_prob
                if i > 0:
                    input_keep_prob = 1
                with tf.variable_scope("biLSTM%dl" % i):
                    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                    输出['lstm_fw_cell%dL' % (i + 1)] = lstm_fw_cell
                    输出['lstm_bw_cell%dL' % (i + 1)] = lstm_bw_cell
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_fw_cell,
                        input_keep_prob=input_keep_prob,
                        output_keep_prob=output_keep_prob,
                    )
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_bw_cell,
                        input_keep_prob=input_keep_prob,
                        output_keep_prob=output_keep_prob,
                    )
                    outputs, output_states_fw, _ = rnn.stack_bidirectional_dynamic_rnn([lstm_fw_cell], [lstm_bw_cell],
                                                                                       outputs, dtype=tf.float32,
                                                                                       sequence_length=sequence_length)
                    前向隐层 = output_states_fw[-1][-1]
                    反向隐层 = tf.split(tf.squeeze(outputs[:, :1, :], [1]), 2, 1)[1]
                    # 每个状态使用 pooling
                    if biLSTM池化方法 != tf.concat:
                        outputs = biLSTM池化方法(tf.split(outputs, 2, 2), 0)
                    # 最后一层
                    if i + 1 >= len(各隐藏层数l):
                        输出['outputs_all'] = outputs
                        # 输出使用 pooling
                        f = eval(模型参数d['LSTM序列池化方法'])
                        if f:
                            if f == tf.reduce_mean:  # padding 输出保证是0
                                outputs = tf.reduce_sum(outputs, 1) / sequence_length
                            elif f == tf.reduce_max:
                                outputs = f(outputs, 1)
                            else:
                                assert False, '不支持其他池化方式!'
                        else:  # 输出最后一个状态
                            if biLSTM池化方法 != tf.concat:
                                outputs = biLSTM池化方法([前向隐层, 反向隐层], 0)
                            else:
                                outputs = tf.concat([前向隐层, 反向隐层], 1)
                        输出['outputs'] = outputs
            if 可视化:
                for name, cell in 输出.items():
                    if 'cell' not in name:
                        continue
                    tf.summary.histogram(name + '-w', cell.weights[0])
                    tf.summary.histogram(name + '-b', cell.weights[1])
            return 输出

        def CNN2d(x, 模型参数d, keep_prob=1., 可视化=True):
            '''
            描述: conv2d + relu + max_pool + concat
            :param x: [batch_size, sequence_length, embedding_dim]
            :param 模型参数d: dict
            :param keep_prob: float
            :param 可视化: bool
            :return:
            '''
            num_filters = 模型参数d['num_filters']
            filter_sizes = 模型参数d['filter_sizes']
            embedding_dim = int(x.shape[2])
            x = tf.expand_dims(x, -1)

            输出 = {}
            with tf.variable_scope("CNN2d"):
                pooled = []
                for filter_size in filter_sizes:
                    filter_shape = [filter_size, embedding_dim, 1, num_filters]

                    # test：从文件读取固定参数检验模型参数问题
                    # if filter_size in {1}:
                    #     print(可视化)
                    #     if 可视化:  # tf 中首次命名出现的参数会保留，而不像torch后初始化的保留
                    #         a = np.swapaxes(np.expand_dims(np.load('../text_representation2/aa_w_f.npy'), -1), 0, -1)
                    #     else:
                    #         a = np.swapaxes(np.expand_dims(np.load('../text_representation2/aa_w_l.npy'), -1), 0, -1)
                    #     w = tf.get_variable(initializer=a, name='CNN_cell%dfs_w' % filter_size)
                    # else:
                    #     w = tf.get_variable(initializer=tf.constant(0.1, shape=filter_shape),
                    #                         name='CNN_cell%dfs_w' % filter_size)
                    # test：用于固定参数检验输出结果
                    # a = tf.constant(0.1, shape=filter_shape)
                    # w = tf.get_variable(initializer=a, name='CNN_cell%dfs_w' % filter_size)
                    # w = tf.get_variable(initializer=a * tf.reshape(tf.linspace(-1., 1., embedding_dim), [1, -1, 1, 1]),
                    #                     name='CNN_cell%dfs_w' % filter_size)

                    # 权重
                    w = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1),
                                        name='CNN_cell%dfs_w' % filter_size)
                    b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_filters]),
                                        name='CNN_cell%dfs_b' % filter_size)
                    输出['CNN_cell%dfs_w' % filter_size] = w
                    输出['CNN_cell%dfs_b' % filter_size] = b
                    # 卷积
                    conv = tf.nn.conv2d(x, w, strides=[1] * 4, padding='VALID')
                    h = tf.nn.relu(tf.nn.bias_add(conv, b))
                    max_pool = tf.nn.max_pool(h, ksize=[1, h.shape[1], 1, 1], strides=[1] * 4, padding='VALID')
                    if 模型参数d['CNN输出层tanh']:
                        max_pool = tf.tanh(max_pool)
                    pooled.append(max_pool)
                num_filters_total = num_filters * len(filter_sizes)
                pooled = tf.reshape(tf.concat(pooled, 3), [-1, num_filters_total])
                pooled = tf.nn.dropout(pooled, keep_prob)
                输出['outputs'] = pooled
            if 可视化:
                for name, cell in 输出.items():
                    if 'cell' not in name:
                        continue
                    tf.summary.histogram(name, cell)
            return 输出

        assert 模型参数d['使用LSTM'] or 模型参数d['使用CNN'], '至少使用一种训练模型!'
        keep_prob_LSTM = tf.cond(isTrain, lambda: 模型参数d['LSTM_dropout'], lambda: 1.)
        keep_prob_CNN = tf.cond(isTrain, lambda: 模型参数d['CNN_dropout'], lambda: 1.)
        输出t = {'outputs_all': title_p_n}
        输出a = {'outputs_all': abstract_p_n}
        if 模型参数d['共享参数']:
            with tf.variable_scope('share_model', reuse=tf.AUTO_REUSE):
                if 模型参数d['使用LSTM']:
                    输出t = biLSTM(title_p_n, title_len, 模型参数d, keep_prob_LSTM)
                    输出a = biLSTM(abstract_p_n, abstract_len, 模型参数d, keep_prob_LSTM, False)
                if 模型参数d['使用CNN']:
                    输出t = CNN2d(输出t['outputs_all'], 模型参数d, keep_prob_CNN)
                    输出a = CNN2d(输出a['outputs_all'], 模型参数d, keep_prob_CNN, False)
        else:
            with tf.variable_scope('title_model', reuse=tf.AUTO_REUSE):
                if 模型参数d['使用LSTM']:
                    输出t = biLSTM(title_p_n, title_len, 模型参数d, keep_prob_LSTM)
                if 模型参数d['使用CNN']:
                    输出t = CNN2d(输出t['outputs_all'], 模型参数d, keep_prob_CNN)
            with tf.variable_scope('abstract_model', reuse=tf.AUTO_REUSE):
                if 模型参数d['使用LSTM']:
                    输出a = biLSTM(abstract_p_n, abstract_len, 模型参数d, keep_prob_LSTM)
                if 模型参数d['使用CNN']:
                    输出a = CNN2d(输出a['outputs_all'], 模型参数d, keep_prob_CNN)
        outputs_t = 输出t['outputs']
        outputs_a = 输出a['outputs']
        return outputs_t, outputs_a

    def _词向量微调层(self, 句子词张量):
        with tf.variable_scope("fine_tune_embedding", reuse=tf.AUTO_REUSE):
            embedding_dim = int(句子词张量.shape[-1])
            weights = tf.get_variable('weights', initializer=tf.random_normal([embedding_dim, embedding_dim]))
            biases = tf.get_variable('biases', initializer=tf.random_normal([embedding_dim]))
            outputs = tf.einsum('ijk,kl->ijl', 句子词张量, weights) + biases
            outputs = tf.tanh(outputs)
            输出 = {}
            输出['weights'] = weights
            输出['biases'] = biases
            输出['outputs'] = outputs
        return 输出

    def 预_编号与填充(self, 句_词变矩阵l, isTitle, 加入新词=True):
        '''
        如果 使用BERT, 将返回可变长度的 句_词矩阵l, 且不转为词序号
        :param 句_词变矩阵l: [[词,..],..]
        :param isTitle: bool
        :param 加入新词: bool
        :return: [[词序号,..],..], [长度1,..], ..
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        assert not isinstance(句_词变矩阵l[0][0], list), '数据格式错误!(句_词变矩阵l[0][0]=%s)' % str(句_词变矩阵l[0][0])
        新词数, 新词s, 加入新词s = 0, set(), set()
        最大长度 = self._模型参数d['title_maxlen'] if isTitle else self._模型参数d['abstract_maxlen']
        长度l = []

        if not self._模型参数d['使用词向量']:
            句_词矩阵l = []
            for 句子 in 句_词变矩阵l:
                句_词矩阵l.append(句子[:最大长度])
                长度l.append(len(句_词矩阵l[-1]))
        else:
            句_词矩阵l = [[0] * 最大长度 for i in range(len(句_词变矩阵l))]
            词数上限 = self._sess.run(self._sess.graph.get_tensor_by_name('word_count_limit:0'))
            for i, 句子 in enumerate(句_词变矩阵l):
                长度 = 0
                for j, 词 in enumerate(句子):
                    if j >= 最大长度:
                        break
                    if 词 in self._词_序号d:
                        句_词矩阵l[i][长度] = self._词_序号d[词]
                        长度 += 1
                    else:
                        新词数 += 1
                        新词s.add(词)
                        if 词数上限 > len(self._词_序号d) and 加入新词 and self._模型参数d['可加入新词']:
                            加入新词s.add(词)
                            序号 = len(self._词_序号d) + 1  # 0号是填充
                            self._词_序号d[词] = 序号
                            句_词矩阵l[i][长度] = 序号
                            长度 += 1
                长度l.append(长度)
        return 句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s

    def 预_编号与填充_批量(self, 多_句_词变矩阵l, isTitle向量, 加入新词=True):
        '''
        如果 使用BERT, 将返回可变长度的 多_句_词矩阵l, 且不转为词序号
        :param 多_句_词变矩阵l: [[[词,..],..],..]
        :param isTitle向量: [bool,..]
        :param 加入新词: bool; 大于词矩阵容纳上限时为True也增加不了新词
        :return:
        '''
        all新词数, all新词s, all加入新词s = 0, set(), set()
        多_句_词矩阵l = []
        多_长度l = []
        for 句_词变矩阵l, isTitle in zip(多_句_词变矩阵l, isTitle向量):
            句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s = self.预_编号与填充(句_词变矩阵l, isTitle, 加入新词)
            多_句_词矩阵l.append(句_词矩阵l)
            多_长度l.append(长度l)
            all新词数 += 新词数
            all新词s |= 新词s
            all加入新词s |= 加入新词s
        return 多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s

    def train(self, title_p, abstract_p, title_n, abstract_n,
              title_len_p, abstract_len_p, title_len_n, abstract_len_n,
              记录过程=True, 记录元数据=False, 合并之前训练错误的数据=None):
        '''
        [title_p, abstract_p], [title_n, abstract_n] 每行要含有相同的标题或摘要, 才能并行训练, 和损失函数有关
        如果不使用词向量, 词序号必须是词向量!
        :param title_p: [[词序号,..],..]
        :param abstract_p: [[词序号,..],..]
        :param title_n: [[词序号,..],..]
        :param abstract_n: [[词序号,..],..]
        :param title_len_p: [长度1,..]
        :param abstract_len_p: [长度1,..]
        :param title_len_n: [长度1,..]
        :param abstract_len_n: [长度1,..]
        :param 记录过程: bool
        :param 记录元数据: bool
        :param 合并之前训练错误的数据: None or [...] 包含和训练集一样的8个元素
        :return: ,.. 训练错误的数据:[...]包含和训练集一样的8个元素
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        assert not isinstance(title_p[0][0], list), '数据格式错误!(title_p[0][0])'
        assert len(title_p) == len(title_len_p), '数据格式错误!(len(title_p)==len(title_len_p))'

        if 合并之前训练错误的数据:
            title_p += 合并之前训练错误的数据[0]
            abstract_p += 合并之前训练错误的数据[1]
            title_n += 合并之前训练错误的数据[2]
            abstract_n += 合并之前训练错误的数据[3]
            title_len_p += 合并之前训练错误的数据[4]
            abstract_len_p += 合并之前训练错误的数据[5]
            title_len_n += 合并之前训练错误的数据[6]
            abstract_len_n += 合并之前训练错误的数据[7]

        if 模型参数d['使用词向量']:  # 如果不使用词向量, 词序号必须是词向量
            title_maxlen = self._模型参数d['title_maxlen']
            abstract_maxlen = self._模型参数d['abstract_maxlen']
            assert len(title_p[0]) == title_maxlen, '标题长度与参数不匹配!(%d==%d)' % (len(title_p[0]), title_maxlen)
            assert len(abstract_n[0]) == abstract_maxlen, '摘要长度与参数不匹配!(%d==%d)' % (len(abstract_n[0]), abstract_maxlen)

        feed_dict = {'title_p:0': title_p, 'abstract_p:0': abstract_p,
                     'title_n:0': title_n, 'abstract_n:0': abstract_n,
                     'title_len_p:0': title_len_p, 'abstract_len_p:0': abstract_len_p,
                     'title_len_n:0': title_len_n, 'abstract_len_n:0': abstract_len_n,
                     'isTrain:0': True}

        self._模型参数d['haveTrainingSteps'] += 1
        # 操作
        训练op = {}
        训练op['loss'] = self._sess.graph.get_tensor_by_name('loss_f/loss_op:0')
        训练op['对错二分'] = self._sess.graph.get_tensor_by_name('acc_f/right_error_list:0')
        训练op['acc'] = self._sess.graph.get_tensor_by_name('acc_f/accuracy_op:0')
        训练op['train'] = self._sess.graph.get_operation_by_name('train_op')
        训练op['loss对错二分'] = self._sess.graph.get_tensor_by_name('loss_f/right_error_list:0')
        # 训练
        if self._可视化w and 记录过程:
            训练op['merged'] = self._sess.graph.get_tensor_by_name('merged_op/merged_op:0')
            if 记录元数据:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            else:
                run_options = None
                run_metadata = None
            训练结果d = self._sess.run(训练op, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            if 记录元数据:
                self._可视化w.add_run_metadata(run_metadata, 'step%d' % self._模型参数d['haveTrainingSteps'])
            self._可视化w.add_summary(训练结果d['merged'], self._模型参数d['haveTrainingSteps'])
        else:
            训练结果d = self._sess.run(训练op, feed_dict=feed_dict)
        # 获得训练错误的数据
        # title_p, abstract_p, title_n, abstract_n,title_len_p, abstract_len_p, title_len_n, abstract_len_n
        训练错误的数据 = [[] for _ in range(8)]
        训练错误的数据_序号 = []
        for i, true in enumerate(训练结果d['对错二分']):
            if true:
                continue
            训练错误的数据[0].append(title_p[i])
            训练错误的数据[1].append(abstract_p[i])
            训练错误的数据[2].append(title_n[i])
            训练错误的数据[3].append(abstract_n[i])
            训练错误的数据[4].append(title_len_p[i])
            训练错误的数据[5].append(abstract_len_p[i])
            训练错误的数据[6].append(title_len_n[i])
            训练错误的数据[7].append(abstract_len_n[i])
            训练错误的数据_序号.append(i)
        # 获得损失函数大于0的数据
        # title_p, abstract_p, title_n, abstract_n,title_len_p, abstract_len_p, title_len_n, abstract_len_n
        损失函数大于0的数据 = [[] for _ in range(8)]
        损失函数大于0的数据_序号 = []
        for i, true in enumerate(训练结果d['loss对错二分']):
            if true:
                continue
            损失函数大于0的数据[0].append(title_p[i])
            损失函数大于0的数据[1].append(abstract_p[i])
            损失函数大于0的数据[2].append(title_n[i])
            损失函数大于0的数据[3].append(abstract_n[i])
            损失函数大于0的数据[4].append(title_len_p[i])
            损失函数大于0的数据[5].append(abstract_len_p[i])
            损失函数大于0的数据[6].append(title_len_n[i])
            损失函数大于0的数据[7].append(abstract_len_n[i])
            损失函数大于0的数据_序号.append(i)

        输出 = {
            '损失函数值': 训练结果d['loss'],
            '精确度': 训练结果d['acc'],
            '训练错误的数据': 训练错误的数据,
            '损失函数大于0的数据': 损失函数大于0的数据,
            '训练错误的数据_序号': 训练错误的数据_序号,
            '损失函数大于0的数据_序号': 损失函数大于0的数据_序号,
            '实际训练数据大小': len(title_p),
        }
        return 输出

    def getTextEmbedding(self, frontPartText_L, backPartText_L, frontPartTextLen_L, backPartTextLen_L, batch_size,
                         l2_normalize):
        '''
        无论参数是否共享, 这里都将使用正例那边的模型
        如果不使用词向量, 词序号必须是词向量!
        :param frontPartText_L: [[词序号,..],..]
        :param backPartText_L: [[词序号,..],..]
        :param frontPartTextLen_L: [长度1,..]
        :param backPartTextLen_L: [长度1,..]
        :param batch_size: int,None, 如果为空则使用全部
        :param l2_normalize: bool, 如果使用这个, 会得到方便cos计算的向量, 用这个向量直接点乘即可
        :return:
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        assert len(frontPartText_L) == len(backPartText_L) == len(frontPartTextLen_L) == len(backPartTextLen_L), '长度不一致'
        # 最大长度
        title_maxlen = self._模型参数d['title_maxlen']
        abstract_maxlen = self._模型参数d['abstract_maxlen']
        # 操作
        向量op = {}
        if l2_normalize:
            向量op['frontText_vec'] = self._sess.graph.get_tensor_by_name('sim_f/title_l2vec:0')
            向量op['backText_vec'] = self._sess.graph.get_tensor_by_name('sim_f/abstract_l2vec:0')
        else:
            向量op['frontText_vec'] = self._sess.graph.get_tensor_by_name('frontText_vec:0')
            向量op['backText_vec'] = self._sess.graph.get_tensor_by_name('backText_vec:0')

        if self._模型参数d['使用词向量']:
            batch_空_f = np.zeros([0, title_maxlen], np.int32)
            batch_空_b = np.zeros([0, abstract_maxlen], np.int32)
        else:
            batch_空_f = np.zeros([0, title_maxlen, self._模型参数d['句矩阵词dim']], np.float32)
            batch_空_b = np.zeros([0, abstract_maxlen, self._模型参数d['句矩阵词dim']], np.float32)

        if batch_size and len(frontPartText_L) > batch_size > 0:  # 如果分批次
            embeddingFront_L = []
            embeddingBack_L = []
            for i in tqdm(range(0, len(frontPartText_L), batch_size), 'getTextEmbedding'):
                batch_f, batch_f_len = frontPartText_L[i:i + batch_size], frontPartTextLen_L[i:i + batch_size]
                batch_b, batch_b_len = backPartText_L[i:i + batch_size], backPartTextLen_L[i:i + batch_size]
                向量 = self._sess.run(向量op,
                                    feed_dict={'title_p:0': batch_f, 'abstract_p:0': batch_b,
                                               'title_n:0': batch_空_f, 'abstract_n:0': batch_空_b,
                                               'title_len_p:0': batch_f_len, 'abstract_len_p:0': batch_b_len,
                                               'title_len_n:0': np.zeros([0], np.int32),
                                               'abstract_len_n:0': np.zeros([0], np.int32),
                                               'isTrain:0': False})
                embeddingFront_L += list(向量['frontText_vec'])
                embeddingBack_L += list(向量['backText_vec'])
        else:
            batch_f, batch_f_len = frontPartText_L, frontPartTextLen_L
            batch_b, batch_b_len = backPartText_L, backPartTextLen_L
            向量 = self._sess.run(向量op,
                                feed_dict={'title_p:0': batch_f, 'abstract_p:0': batch_b,
                                           'title_n:0': batch_空_f, 'abstract_n:0': batch_空_b,
                                           'title_len_p:0': batch_f_len, 'abstract_len_p:0': batch_b_len,
                                           'title_len_n:0': np.zeros([0], np.int32),
                                           'abstract_len_n:0': np.zeros([0], np.int32),
                                           'isTrain:0': False})
            embeddingFront_L = list(向量['frontText_vec'])
            embeddingBack_L = list(向量['backText_vec'])
        # test：输出向量
        # print(np.array(embeddingFront_L))
        # print(np.array(embeddingBack_L))
        return embeddingFront_L, embeddingBack_L

    def saveModel(self, address, save_step=False, max_to_keep=5):
        '''
        新建了Saver就不再用这个地址新建Saver, 定时保存模型会和 _词_序号d, _模型参数d 错位
        :param address: str
        :param save_step: bool 是否保存步数
        :param max_to_keep: int 一个Saver最多保存的模型数
        :return:
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        新建了Saver = False
        if address in self._保存模型地址_saver表:
            saver = self._保存模型地址_saver表[address]
        else:
            with self._sess.graph.as_default():
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                新建了Saver = True
        global_step = self._模型参数d['haveTrainingSteps'] if save_step else None
        saver.save(self._sess, address, global_step=global_step, write_meta_graph=True)
        # 保存其他参数
        global_step = '-' + str(global_step) if global_step else ''
        with open((address + global_step + '.parms').encode('utf-8'), 'w', encoding='utf-8') as w:
            w.write(str(self._模型参数d))
        # 保存词表
        with open((address + global_step + '.w_num_map').encode('utf-8'), 'wb') as w:
            w.write(pickle.dumps(self._词_序号d))
        return 新建了Saver

    def get_parms(self):
        return self._模型参数d

    def close(self):
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        self._sess.close()
        if self._可视化w:
            self._可视化w.close()

    def get_w_embed(self):
        if not self._模型参数d['使用词向量']:
            return
        w_embed = self._sess.graph.get_tensor_by_name('w_embed:0')
        w_embed = self._sess.run(w_embed)
        np.save('av_w_embed.npy', w_embed)
        print('w_embed.shape:', w_embed.shape)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class IRdataSet:
    def __init__(self, 数据集地址, 句子清洗=句子清洗, paperID_probL_idL_L地址=None, 分割位置=None, 句矩阵地址=None):
        self._句子清洗 = 句子清洗
        if paperID_probL_idL_L地址 is None:
            # 用于构建训练集只需要格式: [(paperID,),..]
            # 只用于提取数据(使用平均概率==True or 使用训练集筛选==0, 兼容getTrainSet)只需要格式: [(paperID,[],[]),..]
            # 不使用不平衡数据处理方法只需要格式（weights==[]不会执行随机提取所以没问题）：[(paperID,[],[]),..]
            # 训练集和测试集论文重复应该没事, 这个矩阵只用来构建训练集, 重复的就多训练一次
            self._paperID_probL_noL_L = []  # 在 self._getTextInfor 中赋值, 默认候选集和测试集一起
        else:
            with open(paperID_probL_idL_L地址.encode('utf-8'), 'rb') as r:
                print('读取相似概率矩阵 paperID_probL_idL_L ...')
                self._paperID_probL_noL_L = pickle.load(r)  # [(paperID,[prob,..],[no,..]),..]
        self._trainText1_L, self._trainText2_L, self._trainID_D, self._testText1_L, self._testText2_L, self._testID_L, \
            self._candidateText1_L, self._candidateText2_L, self._candidateID_L, self._allWords_S = self._getTextInfor(
                数据集地址, 分割位置)
        self._trainID_L = [i for i, _ in sorted(self._trainID_D.items(), key=lambda t: t[1])]
        self._分割位置 = 分割位置
        if 句矩阵地址:
            print('句向量存储地址: %s' % 句矩阵地址)
            self._senID_mat_len_seg_mid_h5, self._senID_no_D, self._senMaxLen, self._word_dim, self._senFrontMaxLen, self._senBackMaxLen = self._readSenVectors(
                句矩阵地址)
            print('句矩阵信息= 句子数:%d, 总长度:%d, 词dim:%d, f长度:%d, b长度:%d' %
                  (len(self._senID_no_D), self._senMaxLen, self._word_dim, self._senFrontMaxLen, self._senBackMaxLen))
        else:
            self._senID_mat_len_seg_mid_h5 = None
        print('训练集文本数:%d, 测试集文本数:%d, 候选集文本数:%d' % (
            len(self._paperID_probL_noL_L), len(self._testID_L), len(self._candidateID_L)))

    def getSenMatrix(self, textID_L, all0_front1_back2=0, senFrontLen=None, senBackLen=None):
        """
        获取每个文本的前后部分句子的向量矩阵, 以及句子的长度向量
        """
        assert self._senID_mat_len_seg_mid_h5, '没有句向量矩阵!'
        if not senFrontLen:
            senFrontLen = self._senFrontMaxLen
        if not senBackLen:
            senBackLen = self._senBackMaxLen

        no_S = set([self._senID_no_D[i] for i in textID_L])
        no_L = sorted(no_S)
        no_L_L = [[no_L[0]]]
        for j in no_L[1:]:  # 合并连续的序号
            if no_L_L[-1][-1] == j - 1:
                no_L_L[-1].append(j)
            else:
                no_L_L.append([j])
        matrixs_L_L, lengths_L_L, segPos_L_L, mid_L_L = [], [], [], []
        for j in no_L_L:  # 分批次读取
            matrixs_L_L.append(self._senID_mat_len_seg_mid_h5['matrix'][j])
            lengths_L_L.append(self._senID_mat_len_seg_mid_h5['length'][j])
            segPos_L_L.append(self._senID_mat_len_seg_mid_h5['segPos'][j])
            mid_L_L.append(self._senID_mat_len_seg_mid_h5['mid'][j])
        matrixs_L = np.concatenate(matrixs_L_L, axis=0)
        lengths_L = np.concatenate(lengths_L_L, axis=0)
        segPos_L = np.concatenate(segPos_L_L, axis=0)
        mid_L = np.concatenate(mid_L_L, axis=0)
        no_th_D = {j: k for k, j in enumerate(no_L)}

        senFrontMartix_L = []
        senBackMartix_L = []
        senFrontLen_L = []
        senBackLen_L = []
        for i in textID_L:
            no = no_th_D[self._senID_no_D[i]]
            matrix = matrixs_L[no]
            length = lengths_L[no]
            segPos = segPos_L[no]  # 从0开始, 比word数多1
            mid = mid_L[no]
            if self._分割位置 and 0 < self._分割位置 < 1:
                seg = int(len(segPos) * self._分割位置)
                seg = min(1, seg)  # 不能从0开始
                seg = max(seg, len(segPos) - 1)  # 不能从最后开始
                seg = segPos[seg]
                senFrontMartix_L.append(matrix[: seg])
                senBackMartix_L.append(matrix[seg: length])
            else:
                senFrontMartix_L.append(matrix[: mid])
                senBackMartix_L.append(matrix[mid: length])
            # 长度
            senFrontMartix_L[-1] = senFrontMartix_L[-1][:senFrontLen]
            senBackMartix_L[-1] = senBackMartix_L[-1][:senBackLen]
            senFrontLen_L.append(len(senFrontMartix_L[-1]))
            senBackLen_L.append(len(senBackMartix_L[-1]))
            # 补0
            senFrontMartix_L[-1] = np.concatenate(
                [senFrontMartix_L[-1], np.zeros((senFrontLen - senFrontLen_L[-1], self._word_dim))], axis=0)
            senBackMartix_L[-1] = np.concatenate(
                [senBackMartix_L[-1], np.zeros((senBackLen - senBackLen_L[-1], self._word_dim))], axis=0)
        if all0_front1_back2 == 0:
            return senFrontMartix_L, senFrontLen_L, senBackMartix_L, senBackLen_L
        elif all0_front1_back2 == 1:
            return senFrontMartix_L, senFrontLen_L
        else:
            return senBackMartix_L, senBackLen_L

    @staticmethod
    def _readSenVectors(senMatrixPath):
        senID_mat_len_seg_mid_h5 = h5py.File(senMatrixPath.encode('utf-8'),
                                             'r')  # senID, matrix, length, segPos, mid, senFrontMaxLen, senBackMaxLen
        senID_no_D = {j: i for i, j in enumerate(senID_mat_len_seg_mid_h5['senID'])}
        senMaxLen = senID_mat_len_seg_mid_h5['matrix'].shape[1]
        word_dim = senID_mat_len_seg_mid_h5['matrix'].shape[2]
        senFrontMaxLen = senID_mat_len_seg_mid_h5['senFrontMaxLen'][0]
        senBackMaxLen = senID_mat_len_seg_mid_h5['senBackMaxLen'][0]
        return senID_mat_len_seg_mid_h5, senID_no_D, senMaxLen, word_dim, senFrontMaxLen, senBackMaxLen

    def _getTextInfor(self, 数据集地址, 分割位置=None):
        if self._paperID_probL_noL_L:
            trainID_D = {j[0]: i for i, j in enumerate(self._paperID_probL_noL_L)}  # 编号和位置一一对应
            paperID_probL_noL_L_is_empty = False
        else:
            trainID_D = {}  # {paperID:序号,..}
            paperID_probL_noL_L_is_empty = True
        testID_S = set()
        trainText1_L = [''] * len(trainID_D)  # [[词,..],..]
        trainText2_L = [''] * len(trainID_D)  # [[词,..],..]
        testText1_L = []  # [[词,..],..]
        testText2_L = []  # [[词,..],..]
        testID_L = []  # [文本编号,..]
        trainNum = 0
        candidateText1_L = []
        candidateText2_L = []
        candidateID_L = []  # [文本编号,..]

        allWords_S = set()
        candidatePos_L = []
        testPos_L = []
        allPos_L = []
        
        self.ID_text1_text2_D = {}  # {ID:[句子文本1,句子文本2],..}

        with open(数据集地址.encode('utf-8'), 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '读取检索数据集信息'):
                if i == 0:  # 首行是标签
                    testID_S = set(eval(line.strip()))
                    continue
                line = line.strip().split('\t')
                self.ID_text1_text2_D[line[0]] = [line[1], line[2]]
                if 分割位置 and 0 < 分割位置 < 1:
                    text = self._句子清洗(line[1] + ' ' + line[2]).split(' ')
                    text1, text2 = text[:int(len(text) * 分割位置)], text[int(len(text) * 分割位置):]
                else:
                    text1 = self._句子清洗(line[1]).split(' ')
                    text2 = self._句子清洗(line[2]).split(' ')
                pos = len(text1) / (len(text1) + len(text2))
                allPos_L.append(pos)
                if paperID_probL_noL_L_is_empty:
                    trainID_D[line[0]] = len(trainID_D)
                    self._paperID_probL_noL_L.append((line[0], [], []))
                    trainText1_L.append(text1)
                    trainText2_L.append(text2)
                    trainNum += 1
                elif line[0] in trainID_D:
                    trainText1_L[trainID_D[line[0]]] = text1
                    trainText2_L[trainID_D[line[0]]] = text2
                    trainNum += 1
                if line[0] in testID_S:
                    testText1_L.append(text1)
                    testText2_L.append(text2)
                    testID_L.append(line[0])
                    testPos_L.append(pos)
                else:
                    candidateText1_L.append(text1)
                    candidateText2_L.append(text2)
                    candidateID_L.append(line[0])
                    candidatePos_L.append(pos)
                allWords_S |= set(text1)
                allWords_S |= set(text2)
        candidatePos = sum(candidatePos_L) / len(candidatePos_L)
        testPos = sum(testPos_L) / len(testPos_L)
        allPos = sum(allPos_L) / len(allPos_L)
        print('avg: candidatePos-%f, testPos-%f, allPos-%f' %
              (candidatePos, testPos, allPos))
        assert trainNum == len(trainID_D), '相似概率矩阵与数据集不对应!'
        return trainText1_L, trainText2_L, trainID_D, testText1_L, testText2_L, testID_L, candidateText1_L, candidateText2_L, candidateID_L, allWords_S

    @staticmethod
    def _getTrainSet_singleProcess(args):
        paperID_probL_noL_L, textNum, 使用概率选择的概率, indexL, 使用平均概率 = args
        textID1n, textID2n = [], []
        输出千分比 = 1  # 每输出千分之一输出一次进度
        for i, (_, probL, noL) in enumerate(paperID_probL_noL_L):
            permillage = int(i / len(paperID_probL_noL_L) * 1000)
            if permillage > 输出千分比:
                输出千分比 = permillage
                sys.stdout.write('\r')
                print('构建训练集搭配-%.1f%%---' % (permillage / 10,), end='')
                sys.stdout.flush()

            if random.random() < 使用概率选择的概率:  # 以一定概率
                weights = probL
            else:
                weights = None
            if random.random() < 0.5:  # 负例选择前文本还是后文本
                if weights:
                    if 使用平均概率:
                        textID1n.append(np.random.choice(noL))
                    else:
                        textID1n.append(np.random.choice(noL, p=weights))
                else:  # 如果 weights==[] 自然执行这个
                    textID1n.append(np.random.choice(textNum))
                textID2n.append(indexL + i)
            else:
                textID1n.append(indexL + i)
                if weights:
                    if 使用平均概率:
                        textID2n.append(np.random.choice(noL))
                    else:
                        textID2n.append(np.random.choice(noL, p=weights))
                else:
                    textID2n.append(np.random.choice(textNum))
        sys.stdout.write('\r')
        return textID1n, textID2n  # 存储在训练集中的序号

    def getTrainSet(self, 使用概率选择的概率=0., 进程数=1, 使用平均概率=False, 句向量模式=False):
        if 进程数 > 1:
            avgText = math.ceil(len(self._trainText1_L) / 进程数)
            参数l = [(self._paperID_probL_noL_L[avgText * i: avgText * (i + 1)], len(self._trainText1_L), 使用概率选择的概率,
                    avgText * i, 使用平均概率) for i in range(进程数)]
            pool = Pool(进程数)
            textID1n_textID2n_L = pool.map(self._getTrainSet_singleProcess, 参数l)
            pool.close()
            pool.join()

            textID1n, textID2n = [], []
            for i, j in textID1n_textID2n_L:
                textID1n += i
                textID2n += j
        else:
            textID1n, textID2n = self._getTrainSet_singleProcess(
                (self._paperID_probL_noL_L, len(self._trainText1_L), 使用概率选择的概率, 0, 使用平均概率))

        text1n, text2n = [], []
        if not 句向量模式:
            for i, j in zip(textID1n, textID2n):
                text1n.append(self._trainText1_L[i])
                text2n.append(self._trainText2_L[j])
            trainText1_L = self._trainText1_L
            trainText2_L = self._trainText2_L
        else:
            assert self._senID_mat_len_seg_mid_h5, '没有句矩阵, 不能使用句向量模式!'
            for i, j in zip(textID1n, textID2n):
                text1n.append(self._trainID_L[i])
                text2n.append(self._trainID_L[j])
            trainText1_L = self._trainID_L.copy()
            trainText2_L = trainText1_L.copy()
        return trainText1_L, text1n, trainText2_L, text2n

    @property
    def trainSetSize(self):
        return len(self._trainText1_L)

    def getTestSet(self, 句向量模式=False):
        if not 句向量模式:
            return [self._testText1_L, self._testText2_L], [self._candidateText1_L, self._candidateText2_L]
        else:
            assert self._senID_mat_len_seg_mid_h5, '没有句矩阵, 不能使用句向量模式!'
            return [self._testID_L, self._testID_L], [self._candidateID_L, self._candidateID_L]

    @staticmethod
    def _computeSimMatrix_singleProcess(args):
        testCandidateSimMatrix, topK, index = args
        no_candidate_L = [i for i in range(len(testCandidateSimMatrix[0]))]
        testNo_no8candidate_L = []
        输出千分比 = 1
        for i, sim_L in enumerate(testCandidateSimMatrix):
            permillage = int(i / len(testCandidateSimMatrix) * 1000)
            if permillage > 输出千分比:
                输出千分比 = permillage
                sys.stdout.write('\r')
                print('排序预测结果-%.1f%%---' % (permillage / 10,), end='')
                sys.stdout.flush()

            candidate_sim_L = [(ID, sim) for ID, sim in zip(no_candidate_L, sim_L)]  # 构建相似度列表
            candidate_sim_L = heapq.nlargest(topK, candidate_sim_L, key=lambda t: t[1])  # 排序
            testNo_no8candidate_L.append((i + index, candidate_sim_L))
        sys.stdout.write('\r')
        return testNo_no8candidate_L

    def computeSimMatrixID(self, ftVec_test, btVec_test, ftVec_train, btVec_train, topK, l2_normalize=True, 进程数=1,
                           返回整体矩阵=False):
        sys.stdout.write('\r')
        print('computeSimMatrixID - 相似度计算...', end='')
        sys.stdout.flush()

        if l2_normalize:
            标题_摘要cos矩阵xy = np.dot(ftVec_test, np.transpose(btVec_train))
            标题_摘要cos矩阵yx = np.dot(btVec_test, np.transpose(ftVec_train))
        else:
            标题_摘要cos矩阵xy = np.dot(ftVec_test, np.transpose(btVec_train)) / np.dot(
                np.expand_dims(np.linalg.norm(ftVec_test, axis=1), axis=1),
                np.expand_dims(np.linalg.norm(btVec_train, axis=1), axis=0))
            标题_摘要cos矩阵yx = np.dot(btVec_test, np.transpose(ftVec_train)) / np.dot(
                np.expand_dims(np.linalg.norm(btVec_test, axis=1), axis=1),
                np.expand_dims(np.linalg.norm(ftVec_train, axis=1), axis=0))
        testCandidateSimMatrix = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx]).sum(axis=0)

        # 根据矩阵获得排序好的预测结果
        test_candidate_D = {}  # {测试集论文编号:[候选集论文,..],..}
        if 进程数 > 1:
            avgText = math.ceil(len(testCandidateSimMatrix) / 进程数)
            参数l = [(testCandidateSimMatrix[avgText * i: avgText * (i + 1)], topK, avgText * i) for i in range(进程数)]
            pool = Pool(进程数)
            testNo_no8candidate_L_L = pool.map(self._computeSimMatrix_singleProcess, 参数l)
            pool.close()
            pool.join()
            testNo_no8candidate_L = sum(testNo_no8candidate_L_L, [])
            for no, no_candidate_L in testNo_no8candidate_L:
                test_candidate_D[self._testID_L[no]] = [self._candidateID_L[i] for i, _ in no_candidate_L]
        else:
            for i, sim_L in tqdm(enumerate(testCandidateSimMatrix), '排序预测结果'):
                candidate_sim_L = [(ID, sim) for ID, sim in zip(self._candidateID_L, sim_L)]  # 构建相似度列表
                candidate_sim_L = heapq.nlargest(topK, candidate_sim_L, key=lambda t: t[1])  # 排序
                test_candidate_D[self._testID_L[i]] = [i for i, _ in candidate_sim_L]
        if 返回整体矩阵:
            return test_candidate_D, {
                'testCandidateSimMatrix': testCandidateSimMatrix,
                '标题_摘要cos矩阵xy': 标题_摘要cos矩阵xy,
                '标题_摘要cos矩阵yx': 标题_摘要cos矩阵yx,
            }
        return test_candidate_D

    def getWordEmbedding(self, 词向量地址, 前多少个):
        print('词向量地址: %s' % 词向量地址)
        name = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        sys.stdout.flush()
        词_向量l = []
        vector = []
        with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8', errors='ignore') as r:
            for line in tqdm(r, name):
                line = line.strip().split(' ')
                if len(line) < 3:
                    continue
                word = line[0]
                if word in self._allWords_S:
                    vector = [float(i) for i in line[1:]]
                    词_向量l.append([word, vector])
                    if 前多少个 <= len(词_向量l) or len(self._allWords_S) <= len(词_向量l):
                        break
            维度 = len(vector)
        输出 = {
            'vec': 词_向量l,
            'embedding_dim': 维度,
            'vec_num': len(词_向量l),
        }
        print('...vec_num:%d' % len(词_向量l))
        return 输出


class TCdataSet(IRdataSet):
    def 计算分类数据集距离矩阵(self, ftVec_test, btVec_test, ftVec_train, btVec_train, l2_normalize=True):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        sys.stdout.write('\r')
        print('计算分类数据集距离矩阵 - 相似度计算...', end='')
        sys.stdout.flush()

        if l2_normalize:
            标题_摘要cos矩阵xy = np.dot(ftVec_test, np.transpose(btVec_train))
            标题_摘要cos矩阵yx = np.dot(btVec_test, np.transpose(ftVec_train))
        else:
            标题_摘要cos矩阵xy = np.dot(ftVec_test, np.transpose(btVec_train)) / np.dot(
                np.expand_dims(np.linalg.norm(ftVec_test, axis=1), axis=1),
                np.expand_dims(np.linalg.norm(btVec_train, axis=1), axis=0))
            标题_摘要cos矩阵yx = np.dot(btVec_test, np.transpose(ftVec_train)) / np.dot(
                np.expand_dims(np.linalg.norm(btVec_test, axis=1), axis=1),
                np.expand_dims(np.linalg.norm(ftVec_train, axis=1), axis=0))
        testCandidateSimMatrix = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx]).sum(axis=0)
        test_train_dis_L = -testCandidateSimMatrix  # 相似度越高距离越小

        # 获取标签
        trainLabel_L = []
        testLabel_L = []
        for textID in self._candidateID_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            trainLabel_L.append(label)
        for textID in self._testID_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            testLabel_L.append(label)
        return test_train_dis_L, trainLabel_L, testLabel_L


def 运行():
    ap = 'data/IR arxiv/'  # 修改不同数据集的 word2vec (globe) 结果只需要修改 ap 和 词向量地址
    ap = 'data/CR USPTO/'
    # ap = 'data/CR dblp/'

    # ------训练模型
    模型地址 = ap + 'av_model2/SPM'  # 参考 data/数据介绍.md 进行修改
    batchSize = 200
    进行多少批次 = 10 ** 6
    保存模型 = False

    # ------读取模型还是新建模型
    if os.path.exists(模型地址 + '.parms'):
        print('读取模型')
        模型参数 = 模型地址  # 读取模型
    else:
        print('新建模型')
        模型参数 = 模型参数d  # 新建模型

    # ------避免不平衡数据的技巧
    使用训练集筛选 = 0.  # 可能需要大量内存, 使用tf-idf概率选择负例的概率, 使用平均概率=True会导致从当前进程中的论文中选择负例(类似与使用训练集筛选=0)
    使用训练集筛选_进程数 = 2
    使用平均概率 = True  # 是否不使用tf-idf相似度作为概率选择负例, 使用训练集筛选>0才有效
    重复训练多少上一轮训练错误的数据 = int(batchSize * 0.)
    paperID_probL_idL_L地址 = ap + 'ac_allTtoT100-paperID_probL_noL_L.pkl'
    paperID_probL_idL_L地址 = None

    # ------句矩阵, 和词向量不同时用
    句矩阵文件夹 = r'F:\data\_large_tem_file\CTE/' + ap
    # 句矩阵文件夹 = r'C:\Users\tanshicheng\Downloads\_large_tem_file\CTE/' + ap
    if not os.path.exists(句矩阵文件夹):
        句矩阵文件夹 = ap
    嵌入类型 = DeepMethods.model.roberta_large
    分开训练 = '-'  # 如果有, 用于表示分开训练的词向量, 通常用 - 表示
    句矩阵存储地址 = 句矩阵文件夹 + 'bc_roberta.large_senID_mat_len_seg_mid-.h5'
    # 句矩阵存储地址 = 句矩阵文件夹 + 'bc_' + 嵌入类型 + '_senID_mat_len_seg_mid' + 分开训练 + '.h5'  # token embedding
    # 句矩阵存储地址 = 句矩阵文件夹 + 'bf_randomSent_senID_mat_len_seg_mid.h5'  # sentence embedding
    # 句矩阵存储地址 = 句矩阵文件夹 + 'bf_readSent-skip_senID_mat_len_seg_mid.h5'  # sentence embedding
    句矩阵存储地址 = None  # 配合 模型参数d['使用词向量'] = True, 否则需要 False

    # ------数据集
    数据集模型 = IRdataSet
    # 数据集模型 = TCdataSet
    分割位置 = None
    数据集地址 = ap + 'dataset.text'
    if 数据集模型 == IRdataSet:
        数据集 = 数据集模型(数据集地址=数据集地址, 句子清洗=句子清洗, paperID_probL_idL_L地址=paperID_probL_idL_L地址, 分割位置=分割位置, 句矩阵地址=句矩阵存储地址)
    else:
        数据集 = 数据集模型(数据集地址=数据集地址, 句子清洗=句子清洗, paperID_probL_idL_L地址=paperID_probL_idL_L地址, 分割位置=分割位置, 句矩阵地址=句矩阵存储地址)

    # ------词向量
    if isinstance(模型参数, dict) and 模型参数d['使用词向量']:
        获取前多少个词向量 = 200000
        词向量地址 = 'data/all arxiv/ak_Corpus_vectors.text'
        词向量地址 = 'data/all USPTO/ak_Corpus_vectors.text'
        词向量地址 = 'data/all dblp/ak_Corpus_vectors.text'
        # 词向量地址 = 'data/CR dblp/ak_doc2vecc_wvs.text'  # 非 globe
        # 词向量地址 = 'data/TC IMDB/ak_glove-Corpus_vectors.text'
        词_向量l = 数据集.getWordEmbedding(词向量地址=词向量地址, 前多少个=获取前多少个词向量)['vec']
        # 词_向量l = None  # 可用于随机词向量. 不使用词向量还需要考虑"模型参数d"中的"固定词向量"
    else:
        词_向量l = None

    # ------评估情况
    预测进程数 = 8
    多少批次测试一次模型 = math.ceil(数据集.trainSetSize / batchSize)  # 测试效果最优则保存模型
    # 多少批次测试一次模型 = 1
    测试前进行批次数 = 多少批次测试一次模型  # 如果为 多少批次测试一次模型 上来先测试
    batch_size_测试集 = 2 * batchSize
    if 数据集模型 == IRdataSet:  # ---IR评估
        topN = 20
        评估_class = IR评估(标签地址=数据集地址)
        评估 = 评估_class.评估
    else:  # ---TC评估
        n_neighbors = 9
        # n_neighbors = [3, 6, 9, 12, 15, 18]
        评估_class = TC评估()
        评估 = 评估_class.距离矩阵评估
    has_case_study = True  # False, True

    # ------记录情况
    取消可视化 = True
    多少批次记录一次 = 10
    记录第一批次元数据 = True
    距离可视化参数 = {  # 用于检索而不是分类
        'interval_epochs': 1,  # 多少epoch保存一次
        'results': [],  # 每次的结果: 测试集*候选集 矩阵 然后 摊平成向量
        'x_axis': [],  # 显示在x轴上的名称
        'max_ret': 5,  # 保存几次结果后进行可视化
        'drawn': True,  # 是否已经绘制
    }

    参数文件_obj = 参数文件('av_para.text', {'多少批次测试一次模型': 多少批次测试一次模型})
    模型参数d['句矩阵词dim'] = DeepMethods.model.getWordDim(嵌入类型)
    with 耦合嵌入_tf模型(模型参数=模型参数, 初始词向量l=词_向量l, 可视化地址=模型地址, 取消可视化=取消可视化) as model:
        model.get_w_embed()
        if not model.get_parms()['使用词向量']:
            print(嵌入类型 + '...')
        title_maxlen, abstract_maxlen = model.get_parms()['title_maxlen'], model.get_parms()['abstract_maxlen']

        总批次 = 初始总批次 = model.get_parms()['haveTrainingSteps']
        记录过程 = False
        epoch, batch, bitchNum = 0, 0, 0
        问题p, 问题n, 答案p, 答案n = None, None, None, None
        问题p长度l, 问题n长度l, 答案p长度l, 答案n长度l = None, None, None, None
        loss_all, acc_all = [], []
        训练错误的数据l = []

        目前最好结果 = [0, 0, 0, 0]  # P, R, epoch, batch
        if 'bastMacro-P' in model.get_parms():
            目前最好结果[0] = model.get_parms()['bastMacro-P']

        while True:
            if 总批次 - 进行多少批次 >= 初始总批次:
                break
            if bitchNum % math.ceil(数据集.trainSetSize / batchSize) == 0:
                if model.get_parms()['使用词向量']:
                    问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet(使用概率选择的概率=使用训练集筛选, 进程数=使用训练集筛选_进程数, 使用平均概率=使用平均概率)
                    多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s = model.预_编号与填充_批量([问题p, 问题n, 答案p, 答案n], [1, 1, 0, 0])
                    问题p, 问题n, 答案p, 答案n = 多_句_词矩阵l
                    问题p长度l, 问题n长度l, 答案p长度l, 答案n长度l = 多_长度l
                    print('训练集大小:%d, 新词数:%d, 新词s:%d, 加入新词s:%d' % (len(问题p), all新词数, len(all新词s), len(all加入新词s)))
                else:
                    问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet(使用概率选择的概率=使用训练集筛选, 进程数=使用训练集筛选_进程数, 使用平均概率=使用平均概率, 句向量模式=True)
                epoch += 1
                batch = 0

            for i in tqdm(range(多少批次测试一次模型 - 测试前进行批次数), '训练'):
                if i != 0 and bitchNum % math.ceil(数据集.trainSetSize / batchSize) == 0:
                    break
                if 总批次 % 多少批次记录一次 == 0:
                    记录过程 = True
                # 构建本批次训练集
                frontText_p = 问题p[batch:batch + batchSize]
                backText_p = 答案p[batch:batch + batchSize]
                frontText_n = 问题n[batch:batch + batchSize]
                backText_n = 答案n[batch:batch + batchSize]
                if model.get_parms()['使用词向量']:
                    frontTextLen_p = 问题p长度l[batch:batch + batchSize]
                    backTextLen_p = 答案p长度l[batch:batch + batchSize]
                    frontTextLen_n = 问题n长度l[batch:batch + batchSize]
                    backTextLen_n = 答案n长度l[batch:batch + batchSize]
                else:
                    frontText_p, frontTextLen_p, backText_p, backTextLen_p = 数据集.getSenMatrix(frontText_p,
                                                                                              all0_front1_back2=0,
                                                                                              senFrontLen=title_maxlen,
                                                                                              senBackLen=abstract_maxlen)
                    frontText_n, frontTextLen_n = 数据集.getSenMatrix(frontText_n,
                                                                   all0_front1_back2=1,
                                                                   senFrontLen=title_maxlen,
                                                                   senBackLen=abstract_maxlen)
                    backText_n, backTextLen_n = 数据集.getSenMatrix(backText_n,
                                                                 all0_front1_back2=2,
                                                                 senFrontLen=title_maxlen,
                                                                 senBackLen=abstract_maxlen)
                batch += batchSize
                # 开始训练
                输出 = model.train(title_p=frontText_p, abstract_p=backText_p,
                                 title_n=frontText_n, abstract_n=backText_n,
                                 title_len_p=frontTextLen_p, abstract_len_p=backTextLen_p,
                                 title_len_n=frontTextLen_n, abstract_len_n=backTextLen_n,
                                 记录过程=记录过程, 记录元数据=记录第一批次元数据,
                                 合并之前训练错误的数据=[i[:重复训练多少上一轮训练错误的数据] for i in 训练错误的数据l])
                # 训练错误的数据l = 输出['训练错误的数据']
                训练错误的数据l = 输出['损失函数大于0的数据']
                测试前进行批次数 += 1
                总批次 += 1
                bitchNum += 1
                记录过程 = False
                记录第一批次元数据 = False
                loss_all.append(输出['损失函数值'])
                acc_all.append(输出['精确度'])

            if loss_all:
                print('总批次:%d, epoch:%d, avg-loss:%f, avg-acc:%f' %
                      (总批次, epoch, sum(loss_all) / len(loss_all), sum(acc_all) / len(acc_all)))
                loss_all, acc_all = [], []
            if 多少批次测试一次模型 <= 测试前进行批次数:
                if 数据集模型 == IRdataSet:
                    print('目前最好结果 P: %.4f, R: %.4f (%d-epochs,%d-batch)' % (
                        目前最好结果[0], 目前最好结果[1], 目前最好结果[2], 目前最好结果[3]))
                else:
                    print('目前最好结果 acc: %.4f, error rate: %.4f (%d-epochs,%d-batch)' % (
                        目前最好结果[0], 目前最好结果[1], 目前最好结果[2], 目前最好结果[3]))
                # 准备测试
                (frontTest_L, backTest_L), (frontCandidate_L, backCandidate_L) = 数据集.getTestSet(
                    句向量模式=not model.get_parms()['使用词向量'])
                frontPartText_L = frontTest_L + frontCandidate_L  # 前面是测试集, 后面是候选集
                backPartText_L = backTest_L + backCandidate_L
                embeddingFrontPart_L, embeddingBackPart_L = [], []
                for i in tqdm(range(0, len(frontPartText_L), batch_size_测试集), '测试'):
                    frontText_L = frontPartText_L[i: i + batch_size_测试集]
                    backText_L = backPartText_L[i: i + batch_size_测试集]
                    if model.get_parms()['使用词向量']:
                        多_句_词矩阵l, 多_长度l, _, _, _ = model.预_编号与填充_批量([frontText_L, backText_L], [1, 0], 加入新词=False)
                        frontText_L, backText_L = 多_句_词矩阵l
                        frontTextLen_L, backTextLen_L = 多_长度l
                    else:
                        frontText_L, frontTextLen_L, backText_L, backTextLen_L = 数据集.getSenMatrix(frontText_L,
                                                                                                  all0_front1_back2=0,
                                                                                                  senFrontLen=title_maxlen,
                                                                                                  senBackLen=abstract_maxlen)
                    # test：检查输入的向量
                    # print('frontText_L:', np.array(frontText_L))
                    # print('backText_L:', np.array(backText_L))
                    # 开始测试
                    embeddingFront_L, embeddingBack_L = model.getTextEmbedding(frontText_L, backText_L,
                                                                               frontTextLen_L, backTextLen_L,
                                                                               None, l2_normalize=True)
                    embeddingFrontPart_L += embeddingFront_L
                    embeddingBackPart_L += embeddingBack_L
                # 评估
                if 数据集模型 == IRdataSet:
                    # 是否绘制远离趋势,需要则不适用l2范数,而是余弦相似度
                    if not 距离可视化参数['drawn'] and epoch % 距离可视化参数['interval_epochs'] == 0 and len(距离可视化参数['results']) < 距离可视化参数['max_ret']:
                        l2_normalize = False
                    else:
                        l2_normalize = True
                    test_candidate_D, 矩阵s = 数据集.computeSimMatrixID(
                        ftVec_test=embeddingFrontPart_L[:len(frontTest_L)],
                        btVec_test=embeddingBackPart_L[:len(frontTest_L)],
                        ftVec_train=embeddingFrontPart_L[len(frontTest_L):],
                        btVec_train=embeddingBackPart_L[len(frontTest_L):],
                        topK=topN, l2_normalize=l2_normalize, 进程数=预测进程数, 返回整体矩阵=True)
                    输出 = 评估(test_candidate_D, topN=topN, 简化=True, 输出地址=None)
                    r_a, r_b = 输出['macro-P'], 输出['macro-R']
                    # 远离趋势绘图
                    if not l2_normalize:
                        距离可视化参数['results'].append(矩阵s['标题_摘要cos矩阵yx'].ravel())  # 以后半截为辅助文档向量
                        距离可视化参数['x_axis'].append(f'{epoch}\n%.4f' % r_a)  # 如果上来就测试,实际上epoch应该从0开始
                    if not 距离可视化参数['drawn'] and len(距离可视化参数['results']) == 距离可视化参数['max_ret']:
                        save_path = 'av_trend_chart.pdf'
                        print('绘制远离趋势变化的图...', save_path)
                        draw = Draw(length=5, width=5, r=1, c=1)
                        draw.add_violin(距离可视化参数['results'], 距离可视化参数['x_axis'],
                                        xlabel_name='Epoch\nPrecision', ylabel_name='Cosine similarity')
                        draw.draw(save_path)
                        距离可视化参数['drawn'] = True
                else:
                    test_train_dis_L, trainLabel_L, testLabel_L = 数据集.计算分类数据集距离矩阵(
                        ftVec_test=embeddingFrontPart_L[:len(frontTest_L)],
                        btVec_test=embeddingBackPart_L[:len(frontTest_L)],
                        ftVec_train=embeddingFrontPart_L[len(frontTest_L):],
                        btVec_train=embeddingBackPart_L[len(frontTest_L):],
                        l2_normalize=True)
                    输出 = 评估(test_train_dis_L, trainLabel_L, testLabel_L,
                            n_neighbors=n_neighbors,
                            n_jobs=预测进程数, knn使用距离加权=False, 输出控制台=True)
                    r_a, r_b = 输出['acc'][0], 输出['error rate'][0]
                if r_a >= 目前最好结果[0]:
                    目前最好结果[0] = r_a
                    目前最好结果[1] = r_b
                    目前最好结果[2] = epoch
                    目前最好结果[3] = 总批次
                    model.get_parms()['bastMacro-P'] = 目前最好结果[0]
                    if 保存模型:
                        model.saveModel(模型地址)
                        print('保存了一次模型: %d step' % 总批次)
                    if has_case_study:
                        评估_class.case_study(
                            预测标签=test_candidate_D, 
                            ID_P_R=输出['ID_P_R'], 
                            ID_text1_text2_D=数据集.ID_text1_text2_D, 
                            topN=20, 
                            output_path=模型地址 + '.case_study.json',
                            first_sent='USPTO' in ap
                        )
                # 修改 默认多少批次测试一次模型
                测试前进行批次数 = 0
                x = 参数文件_obj.para['多少批次测试一次模型']
                if x != 多少批次测试一次模型:
                    print('多少批次测试一次模型: %d -> %d' % (多少批次测试一次模型, x))
                    多少批次测试一次模型 = x
                print()


模型参数d = {
    '显存占用比': 0.8,
    'title_maxlen': 200,
    'abstract_maxlen': 200,
    'embedding_dim': 100,
    'margin': 0.1,
    'bastMacro-P': 0.,  # 如果是分类表示的就是acc
    '句矩阵词dim': None,  # 根据不同模型赋值, 构建模型图前赋值

    'learning_rate': 0.3,
    '学习率衰减步数': 1000,
    '学习率衰减率': 0.7,
    '学习率最小值倍数': 100,
    'AdamOptimizer': 0.001,  # 用这个learning_rate和学习率衰减将无效, 非(0,1)则为不使用

    '使用词向量': True,  # 决定是使用词向量还是句矩阵
    '固定词向量': True,  # 决定是否让词向量反向传播
    '词数上限': 200000,  # 不从零开始加一
    '词向量固定值初始化': None,  # 范围在[-1,0) (0,1], 其他值代表随机初始化
    '可加入新词': True,

    '词向量tanh': False,  # 防止词向量爆炸
    '词向量微调': False,

    'LSTM_CNN': {
        'enable': True,
        '共享参数': True,  # 共享标题和摘要模型的参数

        '使用LSTM': False,
        'biLSTM_各隐层数': [200],  # 列表大小为层数
        'biLSTM池化方法': 'tf.concat',
        'LSTM序列池化方法': 'tf.reduce_max',  # 为空表示使用 head/tail
        'LSTM_dropout': 1.,

        '使用CNN': True,  # 使用CNN则 LSTM序列池化方法 无效
        'filter_sizes': [1, 2, 3, 5],
        'num_filters': 1024,
        'CNN_dropout': 1.,
        'CNN输出层tanh': False,
    },
}

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    运行()
