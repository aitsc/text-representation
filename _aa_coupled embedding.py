import os
from tqdm import tqdm
import tensorflow as tf  # 因为 eval 必须 as tf
from tensorflow.contrib import rnn
import random
import sys
import heapq
import pickle
import time
import numpy as np
from bert_serving.client import BertClient
from pprint import pprint
from multiprocessing import Pool
import math
import importlib
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗


class 耦合嵌入_tf模型:
    def __init__(self,模型参数,初始词向量l=None,可视化地址=None,取消可视化=False,BERT句词向量目录=None):
        '''
        词数上限自动加1,从1开始,0是填充
        :param 模型参数d: dict or str 为str表示模型地址,也默认为可视化地址
        :param 初始词向量l: None or [[词,[向量]],..] 为空则随机初始化词向量
        :param 训练可视化地址: str or None
        '''
        assert 模型参数,'缺少模型参数, 是dict或要读取的模型地址.'
        self._可视化w = None
        self._保存模型地址_saver表 = {}
        if isinstance(模型参数,dict):
            assert 'haveTrainingSteps' not in 模型参数,'参数不能包含"haveTrainingSteps"!'
            g, init, self._词_序号d = self._构建计算图(模型参数,初始词向量l)
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数['显存占用比']
            tf_config.gpu_options.allow_growth = True  # GPU按需分配，自适应
            self._sess=tf.Session(graph=g,config=tf_config)
            self._sess.run(init)
            if 可视化地址 and not 取消可视化:
                self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
            self._模型参数d = 模型参数.copy()
            self._模型参数d['haveTrainingSteps']=0
        else:
            self._sess, self._模型参数d, self._词_序号d = self._读取模型(模型参数)
            if not 取消可视化:
                if not 可视化地址: 可视化地址 = 模型参数
                try:
                    self._可视化w = tf.summary.FileWriter(可视化地址, self._sess.graph)
                except:
                    print('模型不含可视化!')
        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            print('开启一个BertClient(需要先开启BertServer)...')
            self._BertClient = BertClient(port=5560, port_out=5561)
            self._BERT句词向量目录 = BERT句词向量目录
            self._BERT句词向量d = {}
            if self._模型参数d['BERT句向量存取加速']: # [str, np.array], ..
                try: # 如果地址存在文件
                    print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + ':读取BERT句向量...', end='')
                    sys.stdout.flush()
                    startTime = time.time()
                    with open(BERT句词向量目录.encode('utf-8'),'rb') as r:
                        while True:
                            try: self._BERT句词向量d.update([pickle.load(r)])
                            except: break
                    print('%.2fm' % ((time.time() - startTime) / 60))
                except:
                    with open(BERT句词向量目录.encode('utf-8'), 'wb'): ...
        else:
            self._BertClient = None

    def _读取模型(self,模型地址):
        '''
        :param 模型地址:
        :return:
        '''
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        startTime=time.time()
        with open((模型地址+'.parms').encode('utf-8'),'r',encoding='utf-8') as r: 模型参数d = eval(r.read())
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.per_process_gpu_memory_fraction = 模型参数d['显存占用比']
        tf_config.gpu_options.allow_growth = True  # GPU按需分配
        sess = tf.Session(config=tf_config)
        saver = tf.train.import_meta_graph(模型地址+'.meta')
        saver.restore(sess,模型地址)
        # 读取词表
        with open((模型地址+'.w_num_map').encode('utf-8'),'rb') as r: 词_序号d = pickle.load(r)
        print('%.2fm' % ((time.time() - startTime) / 60))
        pprint(模型参数d)
        return sess, 模型参数d, 词_序号d

    def _构建计算图(self,模型参数d,初始词向量l):
        '''
        :param 模型参数d: dict 含参数: embedding_dim, title_maxlen, abstract_maxlen, margin, learning_rate, 以及下传model
        :param 初始词向量l: None or [[词,[向量]],..] 为空则随机初始化词向量
        :return: tf.Graph, tf.global_variables_initializer, {词:对应序号,..}
        '''
        print(self.__class__.__name__ + '.' + sys._getframe().f_code.co_name + '...',end='')
        sys.stdout.flush()
        startTime=time.time()
        词_序号d={'':0}
        序号_词d={0:''}
        词向量矩阵l=[[0.]*模型参数d['embedding_dim']]
        graph = tf.Graph()
        with graph.as_default():
            if '使用BERT' not in 模型参数d or not 模型参数d['使用BERT']:
                # 初始化词向量
                词数上限 = 模型参数d['词数上限']
                assert 词数上限,'需要给定词数上限!'
                词数上限 += 1 # 从1开始, 0是填充
                if not 初始词向量l: 初始词向量l = []
                if len(初始词向量l) > 词数上限: 词数上限 = len(初始词向量l) + 1 # 从1开始, 0是填充向量
                for i,(词,向量) in enumerate(初始词向量l):
                    i+=1 # 从1开始, 0是填充
                    词_序号d[词]=i
                    序号_词d[i]=词
                    词向量矩阵l.append(向量)
                for i in range(len(词向量矩阵l),词数上限):
                    if 模型参数d['词向量固定值初始化'] and -1<=模型参数d['词向量固定值初始化']<=1:
                        词向量矩阵l.append([模型参数d['词向量固定值初始化']]*模型参数d['embedding_dim'])
                    else:
                        词向量矩阵l.append([random.uniform(-1, 1) for i in range(模型参数d['embedding_dim'])])
                if 模型参数d['固定词向量']: w_embed = tf.constant(词向量矩阵l, name='w_embed')
                else: w_embed = tf.get_variable('w_embed', initializer=词向量矩阵l)
                tf.constant(词数上限,name='word_count_limit')
                # 词向量信息
                with tf.variable_scope('w_embed_summary'):
                    w_embed_均值 = tf.reduce_mean(w_embed)
                    w_embed_方差 = tf.reduce_mean(tf.square(w_embed - w_embed_均值))
                    w_embed_标准差 = tf.sqrt(w_embed_方差)
                    tf.summary.histogram('w_embed_weights', w_embed)
                    tf.summary.scalar('w_embed_weights_E', w_embed_均值)
                    tf.summary.scalar('w_embed_weights_S', w_embed_标准差)
                # 正例-负例 变量
                title_p = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']],name='title_p')
                abstract_p = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']],name='abstract_p')
                title_n = tf.placeholder(tf.int32, [None, 模型参数d['title_maxlen']],name='title_n')
                abstract_n = tf.placeholder(tf.int32, [None, 模型参数d['abstract_maxlen']],name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)
                # 提供词向量
                title_p_n = tf.nn.embedding_lookup(w_embed, title_p_n)
                abstract_p_n = tf.nn.embedding_lookup(w_embed, abstract_p_n)
            else:
                # 正例-负例 变量
                title_p = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='title_p')
                abstract_p = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='abstract_p')
                title_n = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='title_n')
                abstract_n = tf.placeholder(tf.float32, [None, 模型参数d['BERT_maxlen'], 模型参数d['BERT_embedding_dim']], name='abstract_n')
                title_p_n = tf.concat([title_p, title_n], 0)
                abstract_p_n = tf.concat([abstract_p, abstract_n], 0)

            # 文本实际长度 变量
            title_len_p = tf.placeholder(tf.int32, [None],name='title_len_p')
            abstract_len_p = tf.placeholder(tf.int32, [None],name='abstract_len_p')
            title_len_n = tf.placeholder(tf.int32, [None],name='title_len_n')
            abstract_len_n = tf.placeholder(tf.int32, [None],name='abstract_len_n')
            title_len = tf.concat([title_len_p, title_len_n],0)
            abstract_len = tf.concat([abstract_len_p, abstract_len_n],0)

            # 使用BERT起止符的mask矩阵
            if '使用BERT' in 模型参数d and 模型参数d['使用BERT']:
                if not 模型参数d['使用[SEP]']:
                    title_len -= 1  # [SEP]也充零
                    abstract_len -= 1  # [SEP]也充零
                    mask_t = tf.sequence_mask(title_len, 模型参数d['BERT_maxlen'])
                    mask_a = tf.sequence_mask(abstract_len, 模型参数d['BERT_maxlen'])
                    零一 = tf.constant([[0.]*模型参数d['BERT_embedding_dim'], [1.]*模型参数d['BERT_embedding_dim']])
                    mask_t = tf.nn.embedding_lookup(零一, tf.to_int32(mask_t))
                    mask_a = tf.nn.embedding_lookup(零一, tf.to_int32(mask_a))
                    title_p_n *= mask_t
                    abstract_p_n *= mask_a
                if not 模型参数d['使用[CLS]']:
                    title_len -= 1  # [CLS]充零
                    abstract_len -= 1  # [CLS]充零
                    title_p_n = tf.concat([title_p_n[:,1:,:], title_p_n[:,:1,:]*0], axis=1)
                    abstract_p_n = tf.concat([abstract_p_n[:,1:,:], abstract_p_n[:,:1,:]*0], axis=1)

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
            assert 模型参数d['使用LSTM'] or 模型参数d['使用CNN'], '至少使用一种训练模型!'
            keep_prob_LSTM = tf.placeholder(tf.float32,name='keep_prob_LSTM')
            keep_prob_CNN = tf.placeholder(tf.float32,name='keep_prob_CNN')
            输出t = {'outputs_all': title_p_n}
            输出a = {'outputs_all': abstract_p_n}
            if 模型参数d['共享参数']:
                with tf.variable_scope('share_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出t = self._biLSTM(title_p_n, title_len, 模型参数d,keep_prob_LSTM)
                        输出a = self._biLSTM(abstract_p_n, abstract_len, 模型参数d,keep_prob_LSTM,False)
                    if 模型参数d['使用CNN']:
                        输出t = self._CNN2d(输出t['outputs_all'], 模型参数d,keep_prob_CNN)
                        输出a = self._CNN2d(输出a['outputs_all'], 模型参数d,keep_prob_CNN,False)
            else:
                with tf.variable_scope('title_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出t = self._biLSTM(title_p_n,title_len,模型参数d,keep_prob_LSTM)
                    if 模型参数d['使用CNN']:
                        输出t = self._CNN2d(输出t['outputs_all'], 模型参数d,keep_prob_CNN)
                with tf.variable_scope('abstract_model', reuse=tf.AUTO_REUSE):
                    if 模型参数d['使用LSTM']:
                        输出a =self._biLSTM(abstract_p_n,abstract_len,模型参数d,keep_prob_LSTM)
                    if 模型参数d['使用CNN']:
                        输出a = self._CNN2d(输出a['outputs_all'], 模型参数d,keep_prob_CNN)
            outputs_t = 输出t['outputs']
            outputs_a = 输出a['outputs']

            # 计算距离, 相似度可能为负数
            with tf.variable_scope('sim_f'):
                # GESD, Geometric mean of Euclidean and Sigmoid Dot product
                # sim_p_n = 1 / ((1 + tf.abs(outputs_t - outputs_a)) * (1 + tf.exp(-1-tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1))))
                # AESD, Arithmetic mean of Euclidean and Sigmoid Dot product
                # sim_p_n = 0.5 / (1 + tf.abs(outputs_t - outputs_a)) + 0.5 / (1 + tf.exp(-1-tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1)))
                # 余弦距离, 变这个需要变 标题摘要相似度计算
                outputs_t = tf.nn.l2_normalize(outputs_t, 1, name='title_l2vec')
                outputs_a = tf.nn.l2_normalize(outputs_a, 1, name='abstract_l2vec')
                sim_p_n = tf.reduce_sum(tf.multiply(outputs_t, outputs_a), 1,name='sim_p_n')

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
                loss_op = tf.reduce_sum(tf.maximum(0.0, 负正差),name='loss_op')
                # loss_op = tf.reduce_sum(对数负正差,name='loss_op')
                # loss_op = tf.reduce_sum(tf.exp(负正差 + 一正差)-1, name='loss_op')
                # loss_op = tf.reduce_sum(tf.maximum(0., tf.exp(负正差 + 一正差)-1), name='loss_op')
                tf.summary.scalar('loss', loss_op)
                tf.summary.scalar('loss_max', tf.reduce_max(负正差))
                对错二分_loss = tf.to_float(tf.greater(0., 负正差), name='right_error_list')
            # 准确率
            with tf.variable_scope('acc_f'):
                对错二分 = tf.to_float(tf.greater(sim_p, sim_n), name = 'right_error_list')
                accuracy_op=tf.reduce_mean(对错二分,name='accuracy_op')
                tf.summary.scalar('accuracy', accuracy_op)

            # 指标图
            with tf.variable_scope('index'):
                tf.summary.histogram('P_sub_N_sim', sim_p - sim_n)

            # 梯度下降
            global_step = tf.Variable(0)
            if 0<模型参数d['AdamOptimizer']<1:
                optimizer = tf.train.AdamOptimizer(learning_rate=模型参数d['AdamOptimizer'])
            else:
                学习率最小值 = 模型参数d['learning_rate']/模型参数d['学习率最小值倍数']
                learning_rate = tf.maximum(学习率最小值, tf.train.exponential_decay(
                    learning_rate = 模型参数d['learning_rate'],
                    global_step = global_step,
                    decay_steps = 模型参数d['学习率衰减步数'],
                    decay_rate = 模型参数d['学习率衰减率'],
                ))
                tf.summary.scalar('learning_rate', learning_rate)
                optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
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

        print('%.2fm'%((time.time()-startTime)/60))
        return graph,init,词_序号d

    def _biLSTM(self,x, sequence_length, 模型参数d, keep_prob=1., 可视化=True):
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
        for i,隐层数 in enumerate(各隐藏层数l):
            input_keep_prob = keep_prob
            output_keep_prob = keep_prob
            if i>0:
                input_keep_prob = 1
            with tf.variable_scope("biLSTM%dl"%i):
                lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(隐层数, forget_bias=1.0)
                输出['lstm_fw_cell%dL'%(i+1)] = lstm_fw_cell
                输出['lstm_bw_cell%dL'%(i+1)] = lstm_bw_cell
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_fw_cell,
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                )
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                    lstm_bw_cell,
                    input_keep_prob = input_keep_prob,
                    output_keep_prob = output_keep_prob,
                )
                outputs, output_states_fw, _ = rnn.stack_bidirectional_dynamic_rnn([lstm_fw_cell], [lstm_bw_cell], outputs,dtype=tf.float32,sequence_length=sequence_length)
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
                        if f == tf.reduce_mean: # padding 输出保证是0
                            outputs = tf.reduce_sum(outputs, 1) / sequence_length
                        elif f == tf.reduce_max:
                            outputs = f(outputs, 1)
                        else:
                            assert False,'不支持其他池化方式!'
                    else: # 输出最后一个状态
                        if biLSTM池化方法 != tf.concat:
                            outputs = biLSTM池化方法([前向隐层, 反向隐层], 0)
                        else:
                            outputs = tf.concat([前向隐层, 反向隐层], 1)
                    输出['outputs']=outputs
        if 可视化:
            for name, cell in 输出.items():
                if 'cell' not in name: continue
                tf.summary.histogram(name + '-w', cell.weights[0])
                tf.summary.histogram(name + '-b', cell.weights[1])
        return 输出

    def _CNN2d(self, x, 模型参数d, keep_prob=1., 可视化=True):
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
                # 权重
                w = tf.get_variable(initializer=tf.truncated_normal(filter_shape, stddev=0.1), name='CNN_cell%dfs_b'%filter_size)
                b = tf.get_variable(initializer=tf.constant(0.1, shape=[num_filters]), name='CNN_cell%dfs_w'%filter_size)
                输出['CNN_cell%dfs_w'%filter_size] = w
                输出['CNN_cell%dfs_b'%filter_size] = b
                # 卷积
                conv = tf.nn.conv2d(x, w, strides=[1]*4, padding='VALID')
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
                if 'cell' not in name: continue
                tf.summary.histogram(name, cell)
        return 输出

    def _词向量微调层(self, 句子词张量):
        with tf.variable_scope("fine_tune_embedding",reuse=tf.AUTO_REUSE):
            embedding_dim = int(句子词张量.shape[-1])
            weights = tf.get_variable('weights',initializer=tf.random_normal([embedding_dim, embedding_dim]))
            biases = tf.get_variable('biases',initializer=tf.random_normal([embedding_dim]))
            outputs = tf.einsum('ijk,kl->ijl', 句子词张量, weights) + biases
            outputs = tf.tanh(outputs)
            输出 = {}
            输出['weights'] = weights
            输出['biases'] = biases
            输出['outputs'] = outputs
        return 输出

    def _构建BERT词向量(self,title_p, abstract_p, title_n, abstract_n,
           title_len_p, abstract_len_p, title_len_n, abstract_len_n):
        '''
        注意: title_p, abstract_p, title_n, abstract_n 如果不为空, 需要长度一致
        :param title_p: [[词,..],..]
        :param abstract_p: [[词,..],..]
        :param title_n: [[词,..],..] or []
        :param abstract_n: [[词,..],..] or []
        :param title_len_p: [长度1,..]
        :param abstract_len_p: [长度1,..]
        :param title_len_n: [长度1,..] or []
        :param abstract_len_n: [长度1,..] or []
        :return: 和输入顺序一致, 不过换成np.array格式. 但是如果 title_n 和 title_len_p 为空(必须同时为空), 则不会输出它们, 其他同理
        '''
        assert self._BertClient, '没有BertClient!'
        BERT_maxlen = self._模型参数d['BERT_maxlen']

        # 计算 句子向量张量l = [句子数, BERT_maxlen, BERT_embedding_dim]
        所有句子l = [' '.join(i[:BERT_maxlen]) for i in title_p + abstract_p + title_n + abstract_n]
        if self._模型参数d['BERT句向量存取加速']:
            句子向量张量l = []
            句子序号_位置d = {} # 需要计算的
            待计算句子l = []
            for i,句子 in enumerate(所有句子l):
                if 句子 in self._BERT句词向量d:
                    句子向量张量l.append(self._BERT句词向量d[句子])
                else:
                    句子序号_位置d[len(句子序号_位置d)] = i
                    待计算句子l.append(句子)
                    句子向量张量l.append(None)
            if 待计算句子l:
                计算_句子向量张量l = self._BertClient.encode(待计算句子l)
                w = open(self._BERT句词向量目录.encode('utf-8'), 'ab')
                for i,句子向量张量 in enumerate(计算_句子向量张量l):
                    序号 = 句子序号_位置d[i]
                    句子向量张量l[序号] = 句子向量张量
                    句子 = 所有句子l[序号]
                    self._BERT句词向量d[句子] = 句子向量张量
                    pickle.dump((句子,句子向量张量), w)
                w.close()
            句子向量张量l = np.array(句子向量张量l)
        else:
            句子向量张量l = self._BertClient.encode(所有句子l)

        # 计算 句子长度向量l
        句子长度向量l = title_len_p + abstract_len_p + title_len_n + abstract_len_n
        for i in range(len(句子长度向量l)):  # 不能超过最大长度
            句子长度向量l[i] += 2  # [CLS] 和 [SEP]
            if 句子长度向量l[i] > BERT_maxlen:
                句子长度向量l[i] = BERT_maxlen
        句子长度向量l = np.array(句子长度向量l)

        组数 = bool(title_p) + bool(abstract_p) + bool(title_n) + bool(abstract_n)
        return np.split(句子向量张量l, 组数, axis=0) + np.split(句子长度向量l, 组数, axis=0)

    def 预_编号与填充(self,句_词变矩阵l,isTitle,加入新词=True):
        '''
        如果 使用BERT, 将返回可变长度的 句_词矩阵l, 且不转为词序号
        :param 句_词变矩阵l: [[词,..],..]
        :param isTitle: bool
        :param 加入新词: bool
        :return: [[词序号,..],..], [长度1,..], ..
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        assert not isinstance(句_词变矩阵l[0][0], list), '数据格式错误!(句_词变矩阵l[0][0]=%s)'%str(句_词变矩阵l[0][0])
        新词数, 新词s, 加入新词s = 0, set(), set()
        最大长度 = self._模型参数d['title_maxlen'] if isTitle else self._模型参数d['abstract_maxlen']
        长度l = []

        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            句_词矩阵l = []
            for 句子 in 句_词变矩阵l:
                句_词矩阵l.append(句子[:最大长度])
                长度l.append(len(句_词矩阵l[-1]))
        else:
            句_词矩阵l = [[0]*最大长度 for i in range(len(句_词变矩阵l))]
            词数上限=self._sess.run(self._sess.graph.get_tensor_by_name('word_count_limit:0'))
            for i,句子 in enumerate(句_词变矩阵l):
                长度 = 0
                for j,词 in enumerate(句子):
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
                            序号 = len(self._词_序号d) + 1 # 0号是填充
                            self._词_序号d[词] = 序号
                            句_词矩阵l[i][长度] = 序号
                            长度 += 1
                长度l.append(长度)
        return 句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s

    def 预_编号与填充_批量(self,多_句_词变矩阵l,isTitle向量,加入新词=True):
        '''
        如果 使用BERT, 将返回可变长度的 多_句_词矩阵l, 且不转为词序号
        :param 多_句_词变矩阵l: [[[词,..],..],..]
        :param isTitle向量: [bool,..]
        :param 加入新词: bool
        :return:
        '''
        all新词数, all新词s, all加入新词s = 0, set(), set()
        多_句_词矩阵l = []
        多_长度l = []
        for 句_词变矩阵l,isTitle in zip(多_句_词变矩阵l,isTitle向量):
            句_词矩阵l, 长度l, 新词数, 新词s, 加入新词s = self.预_编号与填充(句_词变矩阵l, isTitle, 加入新词)
            多_句_词矩阵l.append(句_词矩阵l)
            多_长度l.append(长度l)
            all新词数 += 新词数
            all新词s |= 新词s
            all加入新词s |= 加入新词s
        return 多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s

    def 训练(self,title_p, abstract_p, title_n, abstract_n,
           title_len_p, abstract_len_p, title_len_n, abstract_len_n,
           记录过程=True,记录元数据=False,合并之前训练错误的数据=None):
        '''
        [title_p, abstract_p], [title_n, abstract_n] 每行要含有相同的标题或摘要, 才能并行训练, 和损失函数有关
        使用BERT, 词序号也可以是词
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
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        assert not isinstance(title_p[0][0],list),'数据格式错误!(title_p[0][0])'
        assert len(title_p)==len(title_len_p),'数据格式错误!(len(title_p)==len(title_len_p))'

        if 合并之前训练错误的数据:
            title_p += 合并之前训练错误的数据[0]
            abstract_p += 合并之前训练错误的数据[1]
            title_n += 合并之前训练错误的数据[2]
            abstract_n += 合并之前训练错误的数据[3]
            title_len_p += 合并之前训练错误的数据[4]
            abstract_len_p += 合并之前训练错误的数据[5]
            title_len_n += 合并之前训练错误的数据[6]
            abstract_len_n += 合并之前训练错误的数据[7]

        if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
            title_p, abstract_p, title_n, abstract_n, title_len_p, abstract_len_p, title_len_n, abstract_len_n = self._构建BERT词向量(
                title_p, abstract_p, title_n, abstract_n, title_len_p, abstract_len_p, title_len_n, abstract_len_n)
        else:
            title_maxlen = self._模型参数d['title_maxlen']
            abstract_maxlen = self._模型参数d['abstract_maxlen']
            assert len(title_p[0]) == title_maxlen, '标题长度与参数不匹配!(%d==%d)' % (len(title_p[0]), title_maxlen)
            assert len(abstract_n[0]) == abstract_maxlen, '摘要长度与参数不匹配!(%d==%d)' % (len(abstract_n[0]), abstract_maxlen)

        feed_dict = {'title_p:0': title_p, 'abstract_p:0': abstract_p,
                     'title_n:0': title_n, 'abstract_n:0': abstract_n,
                     'title_len_p:0': title_len_p, 'abstract_len_p:0': abstract_len_p,
                     'title_len_n:0': title_len_n, 'abstract_len_n:0': abstract_len_n,
                     'keep_prob_LSTM:0': self._模型参数d['LSTM_dropout'],
                     'keep_prob_CNN:0': self._模型参数d['CNN_dropout']}

        self._模型参数d['haveTrainingSteps'] += 1
        # 操作
        训练op = {}
        训练op['loss']=self._sess.graph.get_tensor_by_name('loss_f/loss_op:0')
        训练op['对错二分']=self._sess.graph.get_tensor_by_name('acc_f/right_error_list:0')
        训练op['acc']=self._sess.graph.get_tensor_by_name('acc_f/accuracy_op:0')
        训练op['train']=self._sess.graph.get_operation_by_name('train_op')
        训练op['loss对错二分']=self._sess.graph.get_tensor_by_name('loss_f/right_error_list:0')
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
        训练错误的数据 = [[] for i in range(8)]
        for i,true in enumerate(训练结果d['对错二分']):
            if true: continue
            训练错误的数据[0].append(title_p[i])
            训练错误的数据[1].append(abstract_p[i])
            训练错误的数据[2].append(title_n[i])
            训练错误的数据[3].append(abstract_n[i])
            训练错误的数据[4].append(title_len_p[i])
            训练错误的数据[5].append(abstract_len_p[i])
            训练错误的数据[6].append(title_len_n[i])
            训练错误的数据[7].append(abstract_len_n[i])
        # 获得损失函数大于0的数据
        # title_p, abstract_p, title_n, abstract_n,title_len_p, abstract_len_p, title_len_n, abstract_len_n
        损失函数大于0的数据 = [[] for i in range(8)]
        for i,true in enumerate(训练结果d['loss对错二分']):
            if true: continue
            损失函数大于0的数据[0].append(title_p[i])
            损失函数大于0的数据[1].append(abstract_p[i])
            损失函数大于0的数据[2].append(title_n[i])
            损失函数大于0的数据[3].append(abstract_n[i])
            损失函数大于0的数据[4].append(title_len_p[i])
            损失函数大于0的数据[5].append(abstract_len_p[i])
            损失函数大于0的数据[6].append(title_len_n[i])
            损失函数大于0的数据[7].append(abstract_len_n[i])

        输出 = {
            '损失函数值':训练结果d['loss'],
            '精确度':训练结果d['acc'],
            '训练错误的数据':训练错误的数据,
            '损失函数大于0的数据':损失函数大于0的数据,
            '实际训练数据大小':len(title_p),
        }
        return 输出

    def 测试(self,批次列表,topN=1,batch_size=None,可加新词=False):
        '''
        相似度可能为负数, 不可以多个标题对多个摘要, 这种方法不可以注意力机制
        :param 批次列表: [[一个标题l,多个摘要l,多个负例l],..] or [[多个标题l,一个摘要l,多个负例l],..]
        :param topN: int 输出每层topN
        :param batch_size: int
        :param 可加新词: bool
        :return: {指标名:[top1结果,..],..}, ..
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        assert not isinstance(批次列表[0][2][0][0],list),'数据格式错误!'
        批次列表_实战=[]
        每批次正例数l=[]
        # 微指标, batch指每个问题或摘要
        batch_top_FP, batch_top_FN, batch_top_TP, batch_top_TN = [], [], [], []
        batch_top_平均相似度 = []

        for j in 批次列表:
            if isinstance(j[0][0], list):
                每批次正例数l.append(len(j[0]))
                标题 = j[0]+j[2]
                摘要 = j[1]
            else:
                每批次正例数l.append(len(j[1]))
                摘要 = j[1]+j[2]
                标题 = j[0]
            批次列表_实战.append([标题,摘要])
        批次_相似度矩阵, all_新词个数, all_新词s, all_加入新词s = self.实战(批次列表=批次列表_实战, batch_size=batch_size, 可加新词=可加新词)
        for i in range(len(批次_相似度矩阵)):
            正例数=每批次正例数l[i]
            序号_得分逆序l = heapq.nlargest(topN, enumerate(批次_相似度矩阵[i]), key=lambda t: t[1])
            预测结果l=[1 if i[0] < 正例数 else 0 for i in 序号_得分逆序l]
            top_FP, top_FN, top_TP, top_TN = [], [], [], []
            top_平均相似度 = []
            预测正确数=0
            for j in range(topN):
                预测正确数+=预测结果l[j]
                top_FP.append(j+1-预测正确数)
                top_FN.append(正例数-预测正确数)
                top_TP.append(预测正确数)
                top_TN.append(len(批次_相似度矩阵[i])-(j+1)-(正例数-预测正确数))
                top_平均相似度.append(sum([k for _,k in 序号_得分逆序l[:j+1]])/(j+1))
            batch_top_FP.append(top_FP)
            batch_top_FN.append(top_FN)
            batch_top_TP.append(top_TP)
            # batch_top_TN.append(top_TN)
            batch_top_平均相似度.append(top_平均相似度)
        batch_top_FP=np.array(batch_top_FP)
        batch_top_FN=np.array(batch_top_FN)
        batch_top_TP=np.array(batch_top_TP)
        # batch_top_TN=np.array(batch_top_TN)
        batch_top_平均相似度=np.array(batch_top_平均相似度)

        top_macroP = (batch_top_TP/(batch_top_TP+batch_top_FP)).mean(axis=0)
        top_macroR = (batch_top_TP/(batch_top_TP+batch_top_FN)).mean(axis=0)
        top_macroF1 = 2*top_macroP*top_macroR/(top_macroP+top_macroR)
        top_平均相似度 = batch_top_平均相似度.mean(axis=0)
        top指标={
            'macro_P':list(top_macroP),
            'macro_R':list(top_macroR),
            'macro_F1':list(top_macroF1),
            '平均相似度':list(top_平均相似度),
        }
        return top指标,all_新词个数, all_新词s, all_加入新词s

    def 实战(self,批次列表,batch_size=None,可加新词=False):
        '''
        相似度可能为负数, 不可以多个标题对多个摘要, 这种方法不可以注意力机制
        :param 批次列表: [[一个标题l,多个摘要l],..] or [[多个标题l,一个摘要l],..]
        :param batch_size: int
        :return: [[一个批次的相似度],..], ..
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0, '还未初始化模型!'
        标题_序号d = {}
        摘要_序号d = {}
        批次列表_序号 = []
        批次_相似度矩阵 = []

        for j in 批次列表:
            # 提取标题和摘要
            if isinstance(j[0][0], list): # 多个标题对一个摘要
                标题l = j[0]
                摘要l = [j[1]]
            else: # 一个标题对多个摘要
                摘要l = j[1]
                标题l = [j[0]]
            # 记录/去重/排序号
            标题序号l = []
            摘要序号l = []
            for 标题 in 标题l:
                标题 = tuple(标题)
                if 标题 not in 标题_序号d:
                    标题_序号d[标题] = len(标题_序号d)
                标题序号l.append(标题_序号d[标题])
            for 摘要 in 摘要l:
                摘要 = tuple(摘要)
                if 摘要 not in 摘要_序号d:
                    摘要_序号d[摘要] = len(摘要_序号d)
                摘要序号l.append(摘要_序号d[摘要])
            # 构建 批次列表_序号
            if isinstance(j[0][0], list): # 多个标题对一个摘要
                批次列表_序号.append([标题序号l, 摘要序号l[0]])
            else: # 一个标题对多个摘要
                批次列表_序号.append([标题序号l[0], 摘要序号l])

        标题l = [i for i,_ in sorted(标题_序号d.items(),key=lambda t:t[1])]
        摘要l = [i for i,_ in sorted(摘要_序号d.items(),key=lambda t:t[1])]
        标题_摘要cos矩阵, all_新词个数, all_新词s, all_加入新词s = self.标题摘要相似度计算(标题l,摘要l,batch_size,可加新词)
        # 相似度还原
        for i in 批次列表_序号:
            批次_相似度矩阵.append([])
            if isinstance(i[0], list): # 多个标题对一个摘要
                摘要序号 = i[1]
                for 标题序号 in i[0]:
                    批次_相似度矩阵[-1].append(标题_摘要cos矩阵[标题序号][摘要序号])
            else: # 一个标题对多个摘要
                标题序号 = i[0]
                for 摘要序号 in i[1]:
                    批次_相似度矩阵[-1].append(标题_摘要cos矩阵[标题序号][摘要序号])
        return 批次_相似度矩阵, all_新词个数, all_新词s, all_加入新词s

    def 标题摘要相似度计算(self,标题l,摘要l,batch_size,可加新词=False,保留进度=True):
        '''
        :param 标题l: 变长矩阵
        :param 摘要l: 变长矩阵
        :param batch_size: int
        :param 可加新词: bool
        :return:
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        # 词向量中未出现的词
        all_新词个数, all_新词s, all_加入新词s = 0, set(), set()
        # 最大长度
        title_maxlen=self._模型参数d['title_maxlen']
        abstract_maxlen=self._模型参数d['abstract_maxlen']
        函数名=self.__class__.__name__ + '.' + sys._getframe().f_code.co_name

        title_p_n, title_len, 新词数, 新词s, 加入新词s = self.预_编号与填充(标题l, True, 加入新词=可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s
        abstract_p_n, abstract_len, 新词数, 新词s, 加入新词s = self.预_编号与填充(摘要l, False, 加入新词=可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s

        feed_dict = {
            'keep_prob_LSTM:0': 1.,
            'keep_prob_CNN:0': 1.
        } # 通用

        标题l2向量op = self._sess.graph.get_tensor_by_name('sim_f/title_l2vec:0')
        标题l2向量l=[]
        for i in tqdm(range(0,len(title_p_n),batch_size), desc=函数名+'(标题)',leave=保留进度):
            batch, batch_len = title_p_n[i:i + batch_size], title_len[i:i + batch_size]
            batch_空 = np.zeros([0, title_maxlen], np.int32)
            if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
                batch, batch_len = self._构建BERT词向量(batch, [], [], [], batch_len, [], [], [])
                batch_空 = np.zeros([0, self._模型参数d['BERT_maxlen'], self._模型参数d['BERT_embedding_dim']], np.float32)
            标题l2向量 = self._sess.run(标题l2向量op,
                                      feed_dict={'title_p:0': batch,
                                                 'title_n:0': batch_空,
                                                 'title_len_p:0': batch_len,
                                                 'title_len_n:0': np.zeros([0], np.int32),
                                                 **feed_dict})
            标题l2向量l+=list(标题l2向量)
        摘要l2向量op = self._sess.graph.get_tensor_by_name('sim_f/abstract_l2vec:0')
        摘要l2向量l=[]

        for i in tqdm(range(0,len(abstract_p_n),batch_size), desc=函数名+'(摘要)',leave=保留进度):
            batch, batch_len = abstract_p_n[i:i + batch_size], abstract_len[i:i + batch_size]
            batch_空 = np.zeros([0, abstract_maxlen], np.int32)
            if '使用BERT' in self._模型参数d and self._模型参数d['使用BERT']:
                batch, batch_len = self._构建BERT词向量(batch, [], [], [], batch_len, [], [], [])
                batch_空 = np.zeros([0, self._模型参数d['BERT_maxlen'], self._模型参数d['BERT_embedding_dim']], np.float32)
            摘要l2向量 = self._sess.run(摘要l2向量op,
                                    feed_dict={'abstract_p:0': batch,
                                               'abstract_n:0': batch_空,
                                               'abstract_len_p:0': batch_len,
                                               'abstract_len_n:0': np.zeros([0], np.int32),
                                               **feed_dict})
            摘要l2向量l+=list(摘要l2向量)
        标题_摘要cos矩阵=np.dot(np.array(标题l2向量l),np.array(摘要l2向量l).T)
        return 标题_摘要cos矩阵, all_新词个数, all_新词s, all_加入新词s

    def 论文相似度计算(self,论文xl,论文yl,batch_size,max0_avg1_min2_sum3_t4_a5=1,可加新词=False):
        '''
        论文xl 和 论文yl 的每篇论文之间的相似度
        :param 论文xl: [[[标题句子1],..],[[摘要句子1],..]]
        :param 论文yl: [[[标题句子1],..],[[摘要句子1],..]]
        :param batch_size: int
        :param max0_avg1_min2_sum3_t4_a5: int 标题和摘要/摘要和标题的相似度的结合方法
        :param 可加新词: bool
        :return:
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        # 词向量中未出现的词
        all_新词个数, all_新词s, all_加入新词s = 0, set(), set()

        标题xl, 摘要xl = 论文xl
        标题yl, 摘要yl = 论文yl
        标题_摘要cos矩阵xy, 新词数, 新词s, 加入新词s = self.标题摘要相似度计算(标题xl, 摘要yl, batch_size,可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s
        标题_摘要cos矩阵yx, 新词数, 新词s, 加入新词s = self.标题摘要相似度计算(标题yl, 摘要xl, batch_size,可加新词)
        all_新词个数 += 新词数
        all_新词s |= 新词s
        all_加入新词s |= 加入新词s

        if max0_avg1_min2_sum3_t4_a5==0:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy,标题_摘要cos矩阵yx.T]).max(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==1:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy,标题_摘要cos矩阵yx.T]).mean(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==2:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx.T]).min(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==3:
            论文cos矩阵xy = np.array([标题_摘要cos矩阵xy, 标题_摘要cos矩阵yx.T]).sum(axis=0)
        elif max0_avg1_min2_sum3_t4_a5==4:
            论文cos矩阵xy = 标题_摘要cos矩阵xy
        else:
            论文cos矩阵xy = 标题_摘要cos矩阵yx.T
        return 论文cos矩阵xy,all_新词个数, all_新词s, all_加入新词s

    def 保存模型(self,address,save_step=False,max_to_keep=5):
        '''
        新建了Saver就不再用这个地址新建Saver, 定时保存模型会和 _词_序号d, _模型参数d 错位
        :param address: str
        :param save_step: bool 是否保存步数
        :param max_to_keep: int 一个Saver最多保存的模型数
        :param keep_checkpoint_every_n_hours: float 隔多少小时会自动保存一次
        :return:
        '''
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        新建了Saver=False
        if address in self._保存模型地址_saver表:
            saver=self._保存模型地址_saver表[address]
        else:
            with self._sess.graph.as_default():
                saver = tf.train.Saver(max_to_keep=max_to_keep)
                新建了Saver=True
        global_step = self._模型参数d['haveTrainingSteps'] if save_step else None
        saver.save(self._sess, address, global_step=global_step, write_meta_graph=True)
        # 保存其他参数
        global_step = '-'+str(global_step) if global_step else ''
        with open((address+global_step+'.parms').encode('utf-8'),'w',encoding='utf-8') as w: w.write(str(self._模型参数d))
        # 保存词表
        with open((address+global_step+'.w_num_map').encode('utf-8'),'wb') as w: w.write(pickle.dumps(self._词_序号d))
        return 新建了Saver

    def get_parms(self):
        return self._模型参数d

    def close(self):
        assert self._模型参数d['haveTrainingSteps'] >= 0,'还未初始化模型!'
        self._sess.close()
        if self._可视化w:
            self._可视化w.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class IRdataSet:
    def __init__(self, 数据集地址, 句子清洗=句子清洗, paperID_probL_idL_L地址=None, 分割位置=None):
        self._句子清洗 = 句子清洗
        with open(paperID_probL_idL_L地址.encode('utf-8'), 'rb') as r:
            print('读取相似概率矩阵 paperID_probL_idL_L ...')
            self._paperID_probL_noL_L = pickle.load(r)  # [(paperID,[prob,..],[no,..]),..]
        self._trainText1_L, self._trainText2_L, self._trainID_D, self._testText1_L, self._testText2_L, self._testID_L, \
        self._candidateText1_L, self._candidateText2_L, self._candidateID_L, self._allWords_S = self._getTextInfor(数据集地址, 分割位置)
        print('训练集文本数:%d, 测试集文本数:%d, 候选集文本数:%d' % (len(self._paperID_probL_noL_L), len(self._testID_L), len(self._candidateID_L)))

    def _getTextInfor(self, 数据集地址, 分割位置=None):
        trainID_D = {j[0]: i for i, j in enumerate(self._paperID_probL_noL_L)}  # 编号和位置一一对应
        testID_S = set()
        trainText1_L = ['']*len(trainID_D)  # [文本1,..]
        trainText2_L = ['']*len(trainID_D)  # [文本2,..]
        testText1_L = []  # [文本1,..]
        testText2_L = []  # [文本2,..]
        testID_L = []  # [文本编号,..]
        trainNum = 0
        candidateText1_L = []
        candidateText2_L = []
        candidateID_L = []  # [文本编号,..]

        allWords_S = set()

        with open(数据集地址.encode('utf-8'), 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '读取检索数据集信息'):
                if i == 0:  # 首行是标签
                    testID_S = set(eval(line.strip()))
                    continue
                line = line.strip().split('\t')
                if 分割位置 and 0 < 分割位置 < 1:
                    text = self._句子清洗(line[1] + ' ' + line[2]).split(' ')
                    text1, text2 = text[:int(len(text)*分割位置)], text[int(len(text)*分割位置):]
                else:
                    text1 = self._句子清洗(line[1]).split(' ')
                    text2 = self._句子清洗(line[2]).split(' ')
                if line[0] in trainID_D:
                    trainText1_L[trainID_D[line[0]]] = text1
                    trainText2_L[trainID_D[line[0]]] = text2
                    trainNum += 1
                if line[0] in testID_S:
                    testText1_L.append(text1)
                    testText2_L.append(text2)
                    testID_L.append(line[0])
                else:
                    candidateText1_L.append(text1)
                    candidateText2_L.append(text2)
                    candidateID_L.append(line[0])
                allWords_S |= set(text1)
                allWords_S |= set(text2)
        assert trainNum == len(trainID_D), '相似概率矩阵与数据集不对应!'
        return trainText1_L, trainText2_L, trainID_D, testText1_L, testText2_L, testID_L, candidateText1_L, candidateText2_L, candidateID_L, allWords_S

    @staticmethod
    def _getTrainSet_singleProcess(args):
        paperID_probL_noL_L, textNum, 使用概率选择的概率, indexL, 使用平均概率 = args
        textID1n, textID2n = [], []
        输出千分比 = 1
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
                else:
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

    def getTrainSet(self, 使用概率选择的概率=0., 进程数=1, 使用平均概率=False):
        if 进程数 > 1:
            avgText = math.ceil(len(self._trainText1_L) / 进程数)
            参数l = [(self._paperID_probL_noL_L[avgText * i: avgText * (i+1)], len(self._trainText1_L), 使用概率选择的概率, avgText * i, 使用平均概率) for i in range(进程数)]
            pool = Pool(进程数)
            textID1n_textID2n_L = pool.map(self._getTrainSet_singleProcess, 参数l)
            pool.close()
            pool.join()

            textID1n, textID2n = [], []
            for i, j in textID1n_textID2n_L:
                textID1n += i
                textID2n += j
        else:
            textID1n, textID2n = self._getTrainSet_singleProcess((self._paperID_probL_noL_L, len(self._trainText1_L), 使用概率选择的概率, 0, 使用平均概率))

        text1n, text2n = [], []
        for i, j in zip(textID1n, textID2n):
            text1n.append(self._trainText1_L[i])
            text2n.append(self._trainText2_L[j])
        return self._trainText1_L, text1n, self._trainText2_L, text2n

    def getTestSet(self):
        return [self._testText1_L, self._testText2_L], [self._candidateText1_L, self._candidateText2_L]

    def computeSimMatrixID(self, testCandidateSimMatrix):  # 根据矩阵获得排序好的预测结果
        test_candidate8sim_D = {}  # {测试集论文编号:[(候选集论文,相似度),..],..}
        test_candidate_D = {}  # {测试集论文编号:[候选集论文,..],..}
        for i, sim_L in tqdm(enumerate(testCandidateSimMatrix), '排序预测结果'):
            test_candidate8sim_D[self._testID_L[i]] = {}
            for ID, sim in zip(self._candidateID_L, sim_L):
                test_candidate8sim_D[self._testID_L[i]][ID] = sim  # {测试集论文编号:{候选集论文:相似度,..},..}
            test_candidate8sim_D[self._testID_L[i]] = sorted(test_candidate8sim_D[self._testID_L[i]].items(), key=lambda t: t[1], reverse=True)
            test_candidate_D[self._testID_L[i]] = [i for i, _ in test_candidate8sim_D[self._testID_L[i]]]
        return test_candidate_D

    def getWordEmbedding(self, 词向量地址, 前多少个):
        name = self.__class__.__name__ + '.' + sys._getframe().f_code.co_name
        sys.stdout.flush()
        词_向量l = []
        vector = []
        with open(词向量地址.encode('utf-8'), 'r', encoding='utf-8') as r:
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


def 运行_IR():
    ap = 'data/'

    batchSize = 250
    epochs = 100000
    多少批次记录一次 = 10
    记录第一批次元数据 = True
    获取前多少个词向量 = 200000
    重复训练多少上一轮训练错误的数据 = int(batchSize*0.0)
    BERT句词向量目录 = ap+'IR arxiv/aa_BERT句向量.pkl'
    词向量地址 = ap+'IR arxiv/ak_arxivCorpus_vectors.text'
    # 词向量地址 = ap+'all arxiv/ak_glove-arxivCorpus_vectors.text'

    使用训练集筛选 = 0.0  # 可能需要大量内存, 以一定概率
    使用训练集筛选_进程数 = 2
    使用平均概率 = True
    取消可视化 = False

    多少轮测试一次模型 = 1  # 测试效果最优则保存模型
    batch_size_测试集 = 500
    目前最好结果 = [0, 0]  # P, epoch
    max0_avg1_min2_sum3_t4_a5 = 3
    # 评估结果输出地址 = ap+'aa_RAP结果.txt'
    评估结果输出地址 = None
    topN = 20

    模型地址 = ap+'IR arxiv/aa_model4/SPM'
    数据集 = IRdataSet(数据集地址=ap+'IR arxiv/IR arxiv.text', 句子清洗=句子清洗,
                    paperID_probL_idL_L地址=ap+'IR arxiv/ac_allTtoT100-paperID_probL_noL_L.pkl',
                    # 相似概率矩阵地址=ap+'af_IR arxiv所有论文BERT距离.pkl',
                    # 相似概率矩阵文本编号地址=ap+'af_IR arxiv所有论文BERT距离-编号.text',
                    分割位置=None)
    IR评估 = importlib.import_module('_ad_evaluate').IR评估
    评估 = IR评估(标签地址=ap+'IR arxiv/IR arxiv.text')

    # 词_向量l = 数据集.getWordEmbedding(词向量地址=词向量地址, 前多少个=获取前多少个词向量)['vec']
    词_向量l = None
    训练错误的数据l = []
    with 耦合嵌入_tf模型(模型参数d,初始词向量l=词_向量l,可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录, 取消可视化=取消可视化) as model:
    # with 耦合嵌入_tf模型(模型参数d, 可视化地址=模型地址, BERT句词向量目录=BERT句词向量目录, 取消可视化=取消可视化) as model:
    # with 耦合嵌入_tf模型(模型参数=模型地址, BERT句词向量目录=BERT句词向量目录, 取消可视化=取消可视化) as model:
        总批次 = model.get_parms()['haveTrainingSteps']
        记录过程 = False
        for epoch in range(epochs):
            问题p, 问题n, 答案p, 答案n = 数据集.getTrainSet(使用概率选择的概率=使用训练集筛选, 进程数=使用训练集筛选_进程数, 使用平均概率=使用平均概率)
            多_句_词矩阵l, 多_长度l, all新词数, all新词s, all加入新词s = model.预_编号与填充_批量([问题p, 问题n, 答案p, 答案n],[1,1,0,0])
            问题p, 问题n, 答案p, 答案n = 多_句_词矩阵l
            问题p长度l, 问题n长度l, 答案p长度l, 答案n长度l = 多_长度l
            print('训练集大小:%d, 新词数:%d, 新词s:%d, 加入新词s:%d' % (len(问题p), all新词数, len(all新词s), len(all加入新词s)))

            loss_all, acc_all = [], []
            for batch in tqdm(range(0, len(问题p), batchSize)):
                if 总批次 % 多少批次记录一次 == 0:
                    记录过程 = True
                输出 = model.训练(title_p=问题p[batch:batch + batchSize], abstract_p=答案p[batch:batch + batchSize],
                              title_n=问题n[batch:batch + batchSize], abstract_n=答案n[batch:batch + batchSize],
                              title_len_p=问题p长度l[batch:batch + batchSize],
                              abstract_len_p=答案p长度l[batch:batch + batchSize],
                              title_len_n=问题n长度l[batch:batch + batchSize],
                              abstract_len_n=答案n长度l[batch:batch + batchSize],
                              记录过程=记录过程,
                              记录元数据=记录第一批次元数据,
                              合并之前训练错误的数据=[i[:重复训练多少上一轮训练错误的数据] for i in 训练错误的数据l])
                # 训练错误的数据l = 输出['训练错误的数据']
                训练错误的数据l = 输出['损失函数大于0的数据']
                总批次 += 1
                记录过程 = False
                记录第一批次元数据 = False
                loss_all.append(输出['损失函数值'])
                acc_all.append(输出['精确度'])
            print('总批次:%d, epoch:%d, loss:%f, acc:%f' %
                  (总批次, epoch + 1, sum(loss_all) / len(loss_all), sum(acc_all) / len(acc_all)))
            print()
            if epoch%多少轮测试一次模型==0:
                测试文本, 候选文本 = 数据集.getTestSet()
                print('目前最好结果P:%.4f(%d-epochs)'%(目前最好结果[0],目前最好结果[1]))
                论文cos矩阵xy, all_新词个数, all_新词s, all_加入新词s = model.论文相似度计算(
                                        论文xl=测试文本,
                                        论文yl=候选文本,
                                        batch_size=batch_size_测试集,
                                        max0_avg1_min2_sum3_t4_a5=max0_avg1_min2_sum3_t4_a5)
                预测标签 = 数据集.computeSimMatrixID(testCandidateSimMatrix=论文cos矩阵xy.tolist())
                输出 = 评估.评估(预测标签, topN=topN, 简化=True, 输出地址=评估结果输出地址)
                if 输出['macro-P'] >= 目前最好结果[0]:
                    目前最好结果[0] = 输出['macro-P']
                    目前最好结果[1] = epoch+1
                    model.保存模型(模型地址)
                    print('保存了一次模型: %d step' % 总批次)


模型参数d={
    '显存占用比': 0.8,
    'title_maxlen': 200,
    'abstract_maxlen': 200,
    'embedding_dim':100,
    'margin': 0.1,
    '共享参数': True, # 共享标题模型和摘要模型的参数

    'learning_rate': 0.3,
    '学习率衰减步数': 1000,
    '学习率衰减率': 0.7,
    '学习率最小值倍数': 100,
    'AdamOptimizer': 0.001, # 用这个learning_rate和学习率衰减将无效, 非(0,1)则为不使用

    '词数上限': 200000, # 不从零开始加一
    '词向量固定值初始化': None, # 范围在[-1,0) (0,1], 其他值代表随机初始化
    '固定词向量': True, # 决定是否让词向量反向传播
    '词向量微调': False,
    '可加入新词': True,
    '词向量tanh': False, # 防止词向量爆炸

    '使用LSTM': False,
    'biLSTM_各隐层数':[200], # 列表大小为层数
    'biLSTM池化方法': 'tf.concat',
    'LSTM序列池化方法': 'tf.reduce_max', # 为空表示使用 head/tail
    'LSTM_dropout': 1.,

    '使用CNN': True, # 使用CNN则 LSTM序列池化方法 无效
    'filter_sizes': [1,2,3,5],
    'num_filters': 1024,
    'CNN_dropout': 1.,
    'CNN输出层tanh': False,

    '使用BERT': True, # 不能用句向量(用句词向量). 不能单独使用(要使用其他模型). 将不在单独生成词向量
    'BERT_maxlen': 200,
    'BERT_embedding_dim': 1024, # embedding_dim 将无效
    '使用[CLS]': True,
    '使用[SEP]': True, # 不使用则[SEP]及后面都充零, 分词不对[SEP]充零会误前移
    'BERT句向量存取加速': True, # 没有地址参数无用. 会占用大量内存和外存(比如一万句10GB), 速度提升10倍
}

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    运行_IR()
    # 运行_IR测试()

r'''
使用BERT前打开服务,比如执行命令: 
bert-serving-start -model_dir /home/tansc/python/fork/uncased_L-24_H-1024_A-16 -gpu_memory_fraction=0.2 -max_seq_len=200 -max_batch_size=64 -num_worker=1 -pooling_strategy=NONE -port 5560 -port_out 5561
bert-serving-start -model_dir D:\data\code\python\GPU-31-11\fork\uncased_L-24_H-1024_A-16 -gpu_memory_fraction=0.2 -max_seq_len=200 -max_batch_size=64 -num_worker=1 -pooling_strategy=NONE -port 5560 -port_out 5561
'''
