import torch  # 1.1.0
from pytorch_transformers import *  # 1.0.0
from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
import time
from tqdm import tqdm
import h5py
import os
import importlib
import logging
logging.basicConfig(level=logging.ERROR)  # 防止输出初始化句子过长的警告
IR评估 = importlib.import_module('_ad_evaluate').IR评估
句子清洗 = importlib.import_module('_au_text preprocessing').句子清洗
TC评估 = importlib.import_module('_ad_evaluate').TC评估


class DeepMethods:
    class model:
        xlnet_large_cased = 'xlnet-large-cased'
        gpt2_medium = 'gpt2-medium'
        openai_gpt = 'openai-gpt'
        transfo_xl_wt103 = 'transfo-xl-wt103'
        xlm_mlm_en_2048 = 'xlm-mlm-en-2048'
        bert_large_uncased = 'bert-large-uncased'
        bert_large_uncased_wwm = 'bert-large-uncased-whole-word-masking'
        ELMo = 'ELMo'
        ELMo_cascade = 'ELMo-cascade'
        roberta_large = 'roberta.large'
        skip_thoughts = 'skip-thoughts'
        dim100_sent = 'dim100_sent'

        @staticmethod
        def getWordDim(modelName):
            if modelName == DeepMethods.model.xlnet_large_cased:
                word_dim = 1024
            elif modelName == DeepMethods.model.gpt2_medium:
                word_dim = 1024
            elif modelName == DeepMethods.model.openai_gpt:
                word_dim = 768
            elif modelName == DeepMethods.model.transfo_xl_wt103:
                word_dim = 1024
            elif modelName == DeepMethods.model.xlm_mlm_en_2048:
                word_dim = 2048
            elif modelName == DeepMethods.model.bert_large_uncased:
                word_dim = 1024
            elif modelName == DeepMethods.model.bert_large_uncased_wwm:
                word_dim = 1024
            elif modelName == DeepMethods.model.ELMo:
                word_dim = 1024
            elif modelName == DeepMethods.model.ELMo_cascade:
                word_dim = 3072
            elif modelName == DeepMethods.model.roberta_large:
                word_dim = 1024
            elif modelName == DeepMethods.model.skip_thoughts:
                word_dim = 4800
            elif modelName == DeepMethods.model.dim100_sent:
                word_dim = 100
            else:
                raise Exception("无效的模型名称!", modelName)
            return word_dim

    def __init__(self, datasetPath, bitchSize, GPU=True, senFrontMaxLen=200, senBackMaxLen=200, modelName=None, senMatrixPath=None, model_path=None, segTrain=False):
        assert senFrontMaxLen, 'senFrontMaxLen 不能为空'
        assert senBackMaxLen, 'senBackMaxLen 不能为空'
        print('数据: ' + datasetPath)
        print('模型: ' + modelName)

        self._candTextF_sentence_L, self._candTextB_sentence_L, self._testTextF_sentence_L, self._testTextB_sentence_L = self._getTextInfor(datasetPath)
        self._bitchSize = bitchSize
        self._GPU = GPU
        self._senFrontMaxLen = senFrontMaxLen
        self._senBackMaxLen = senBackMaxLen
        self._senMaxLen = senFrontMaxLen + senBackMaxLen
        self._segTrain = segTrain

        self._pretrained_model = modelName
        if modelName == self.model.xlnet_large_cased:
            self._getSenVectorsModel = self._modelXLNet
        elif modelName == self.model.gpt2_medium:
            self._getSenVectorsModel = self._modelGPT2
        elif modelName == self.model.openai_gpt:
            self._getSenVectorsModel = self._modelGPT
        elif modelName == self.model.transfo_xl_wt103:
            self._getSenVectorsModel = self._modelTransfoXL
        elif modelName == self.model.xlm_mlm_en_2048:
            self._getSenVectorsModel = self._modelXLM
        elif modelName == self.model.bert_large_uncased or modelName == self.model.bert_large_uncased_wwm:
            self._getSenVectorsModel = self._modelBERT
        elif modelName == self.model.roberta_large:
            self._getSenVectorsModel = self._modelRoBERTa
        elif modelName == self.model.ELMo or modelName == self.model.ELMo_cascade:  # 自带 mask
            self._getSenVectorsModel = self._modelELMo
            self._pretrained_model = model_path
        else:
            raise Exception("无效的模型名称!", modelName)
        self._modelName = modelName
        self._word_dim = self.model.getWordDim(modelName)
        self._model = None
        self._senID_mat_len_seg_mid_h5, self._senID_no_D = self._readSenVectors(senMatrixPath, self._word_dim, senFrontMaxLen, senBackMaxLen)

        self._candSenVec, self._testSenVec = self._getAvgWordEmbedding()

    @staticmethod
    def _readSenVectors(senMatrixPath, word_dim, senFrontMaxLen, senBackMaxLen):
        if not senMatrixPath:
            return None, None
        try:
            senID_mat_len_seg_mid_h5 = h5py.File(senMatrixPath.encode('utf-8'), 'a')
        except:
            print('追加失败, 写入文件')
            senID_mat_len_seg_mid_h5 = h5py.File(senMatrixPath.encode('utf-8'), 'w')
        senMaxLen = senFrontMaxLen + senBackMaxLen
        if 'senID' in senID_mat_len_seg_mid_h5:
            senID_no_D = {j: i for i, j in enumerate(senID_mat_len_seg_mid_h5['senID']) if j}
            assert senID_mat_len_seg_mid_h5['matrix'].shape[1] == senMaxLen, '句矩阵与模型句子长度不一致!'
            assert senID_mat_len_seg_mid_h5['matrix'].shape[2] == word_dim, '句矩阵与模型词维度不一致!'
            assert senID_mat_len_seg_mid_h5['senFrontMaxLen'][0] == senFrontMaxLen, '句矩阵与模型senFrontMaxLen不一致!'
            assert senID_mat_len_seg_mid_h5['senBackMaxLen'][0] == senBackMaxLen, '句矩阵与模型senBackMaxLen不一致!'
        else:
            senID_no_D = {}
            senID_mat_len_seg_mid_h5.create_dataset("senID", (0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=str))
            senID_mat_len_seg_mid_h5.create_dataset("matrix", (0, senMaxLen, word_dim), maxshape=(None, senMaxLen, word_dim), dtype=np.float32, chunks=(1, senMaxLen, word_dim))
            senID_mat_len_seg_mid_h5.create_dataset("length", (0,), maxshape=(None,), dtype=np.int32, chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("segPos", shape=(0,), maxshape=(None,), dtype=h5py.special_dtype(vlen=np.dtype('float32')), chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("mid", (0,), maxshape=(None,), dtype=np.int32, chunks=(1,))
            senID_mat_len_seg_mid_h5.create_dataset("senFrontMaxLen", data=[senFrontMaxLen])
            senID_mat_len_seg_mid_h5.create_dataset("senBackMaxLen", data=[senBackMaxLen])
        return senID_mat_len_seg_mid_h5, senID_no_D

    def _modelXLNet(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = XLNetTokenizer.from_pretrained(self._pretrained_model)
            self._model = XLNetModel.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置
        sentences_mask_L = []
        mid_L = []
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
            # 使用掩码将多余词屏蔽
            hidden_states = self._model(tokens_tensor, attention_mask=mask_tensor)[0]
            senMatrix = hidden_states * torch.unsqueeze(mask_tensor, 2)
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelGPT2(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = GPT2Tokenizer.from_pretrained(self._pretrained_model)
            self._model = GPT2Model.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置
        sentences_mask_L = []
        mid_L = []
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
            hidden_states = self._model(tokens_tensor)[0]
            # 使用掩码将多余词屏蔽
            senMatrix = hidden_states * torch.unsqueeze(mask_tensor, 2)
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelGPT(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = OpenAIGPTTokenizer.from_pretrained(self._pretrained_model)
            self._model = OpenAIGPTModel.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置
        sentences_mask_L = []
        mid_L = []
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
            hidden_states = self._model(tokens_tensor)[-1]
            # 使用掩码将多余词屏蔽
            senMatrix = hidden_states * torch.unsqueeze(mask_tensor, 2)
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelTransfoXL(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = TransfoXLTokenizer.from_pretrained(self._pretrained_model)
            self._model = TransfoXLModel.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置
        sentences_mask_L = []
        mid_L = []
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
            # causal model, 充0等价掩码
            hidden_states = self._model(tokens_tensor)[0]
            senMatrix = hidden_states * torch.unsqueeze(mask_tensor, 2)
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelXLM(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = XLMTokenizer.from_pretrained(self._pretrained_model)
            self._model = XLMModel.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置
        sentences_mask_L = []
        mid_L = []
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen:
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
            # 使用掩码将多余词屏蔽
            hidden_states = self._model(tokens_tensor, attention_mask=mask_tensor)[0]
            senMatrix = hidden_states * torch.unsqueeze(mask_tensor, 2)
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelBERT(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._tokenizer = BertTokenizer.from_pretrained(self._pretrained_model)
            self._model = BertModel.from_pretrained(self._pretrained_model)
            self._model.eval()
            if self._GPU:
                self._model.to('cuda')

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置, 这里取消包括 [CLS]
        sentences_mask_L = []
        sentences_maskAll_L = []  # 将 [CLS] 和 [SEP] 屏蔽, 用于分段. 如果中间有 [SEP] 需要额外考虑了
        mid_L = []
        SEP = self._tokenizer.convert_tokens_to_ids('[SEP]')
        CLS = self._tokenizer.convert_tokens_to_ids('[CLS]')
        if self._senBackMaxLen == 0:
            x = 2
        else:
            x = 1
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen - x:  # 预留 [CLS] or [SEP]
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        if self._senFrontMaxLen == 0:
            x = 2
        else:
            x = 1
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer.encode(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen - x:  # 预留 [SEP]
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            # CLS SEP
            sentences_L[i] = [CLS] + sentences_L[i] + [SEP]
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [0] * m  # 补充掩码
            sentences_mask_L.append([1.] * (self._senMaxLen - m) + [0.] * m)
            sentences_maskAll_L.append([0.] + [1.] * (self._senMaxLen - m - 2) + [0.] * (m + 1))

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            mask_tensor = torch.tensor(sentences_mask_L)
            maskALL_tensor = torch.tensor(sentences_maskAll_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                mask_tensor = mask_tensor.to('cuda')
                maskALL_tensor = maskALL_tensor.to('cuda')
            # 使用掩码将多余词屏蔽
            hidden_states = self._model(tokens_tensor, attention_mask=mask_tensor)[0]
            senMatrix = hidden_states * torch.unsqueeze(maskALL_tensor, 2)
            senMatrix = torch.cat([senMatrix[:, 1:, :], senMatrix[:, :1, :]], dim=1)  # 移除 [CLS]
            senMatrix = np.array(senMatrix.cpu())
        for i in range(len(sentences_mask_L)):  # 移除 [CLS] [SEP]
            sentences_mask_L[i] = sentences_mask_L[i][2:] + [0, 0]
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelELMo(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._model = ElmoEmbedder(self._pretrained_model + 'options.json', self._pretrained_model + 'weights.hdf5', cuda_device=0 if self._GPU else -1)
            try:
                self._model.batch_to_embeddings([['a']])
            except:
                print('停用torch的cuDNN')
                torch.backends.cudnn.enabled = False

        senF_L = [i[:self._senFrontMaxLen] for i in senF_L]
        senB_L = [i[:self._senBackMaxLen] for i in senB_L]
        segPos_L = []
        sentences_L = []
        for f, b in zip(senF_L, senB_L):
            segPos_L.append([i for i in range(len(f) + len(b) + 1)])
            sentences_L.append(f + b)
        mid_L = [len(i) for i in senF_L]

        elmo_embedding, elmo_mask = self._model.batch_to_embeddings(sentences_L)
        # 自带掩码屏蔽
        if self._modelName != self.model.ELMo_cascade:
            senMatrix = np.sum(np.array(elmo_embedding.cpu()), axis=1) / elmo_embedding.shape[1]
        else:  # 如果是级联
            senMatrix = np.concatenate([i.squeeze().cpu() for i in elmo_embedding.chunk(3, 1)], axis=2)
        elmo_mask = np.array(elmo_mask.cpu())
        # 补0
        if senMatrix.shape[1] < self._senMaxLen:
            senMatrix = np.concatenate([senMatrix, np.zeros((senMatrix.shape[0], self._senMaxLen - senMatrix.shape[1], senMatrix.shape[2]))], axis=1)
            elmo_mask = np.concatenate([elmo_mask, np.zeros((elmo_mask.shape[0], self._senMaxLen - elmo_mask.shape[1]))], axis=1)
        return senMatrix, elmo_mask, np.array(segPos_L), np.array(mid_L)  # np.array

    def _modelRoBERTa(self, senF_L, senB_L):
        '''
        :param senF_L: [[词,..],..]
        :param senB_L:  [[词,..],..]
        :return:
        '''
        if not self._model:
            self._model = torch.hub.load('pytorch/fairseq', self._pretrained_model)
            self._tokenizer = lambda x: self._model.encode(x).tolist()[1:-1]  # tensor->list, 去除[CLS] [SEP]
            self._model.eval()
            if self._GPU:
                self._model.cuda()

        sentences_L = [[] for _ in range(len(senF_L))]
        segPos_L = [[0] for _ in range(len(senF_L))]  # 每个词的位置, 这里取消包括 [CLS]
        sentences_mask_L = []
        sentences_maskAll_L = []  # 将 [CLS] 和 [SEP] 屏蔽, 用于分段. 如果中间有 [SEP] 需要额外考虑了
        mid_L = []
        CLS, SEP = self._model.encode('').tolist()  # [CLS] [SEP] token
        if self._senBackMaxLen == 0:
            x = 2
        else:
            x = 1
        for i, sen in enumerate(senF_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senFrontMaxLen - x:  # 预留 [CLS] or [SEP]
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            mid_L.append(len(sentences_L[i]))
        if self._senFrontMaxLen == 0:
            x = 2
        else:
            x = 1
        for i, sen in enumerate(senB_L):
            indexed_tokens_L = []
            for word in sen:
                tokens = self._tokenizer(word)
                seg = segPos_L[i][-1] + len(tokens)
                if seg > self._senBackMaxLen - x:  # 预留 [SEP]
                    break
                indexed_tokens_L.append(tokens)
                segPos_L[i].append(seg)
            indexed_tokens = sum(indexed_tokens_L, [])  # 获得编号
            sentences_L[i] += indexed_tokens
            # CLS SEP
            sentences_L[i] = [CLS] + sentences_L[i] + [SEP]
            m = self._senMaxLen - len(sentences_L[i])  # 掩码长度
            sentences_L[i] += [1] * m  # 补充[PAD] 1
            sentences_mask_L.append([1.] * (self._senMaxLen - m - 2) + [0.] * (m + 2))  # 移除 [CLS] [SEP]
            sentences_maskAll_L.append([0.] + [1.] * (self._senMaxLen - m - 2) + [0.] * (m + 1))

        with torch.no_grad():
            tokens_tensor = torch.tensor(sentences_L)
            maskALL_tensor = torch.tensor(sentences_maskAll_L)
            if self._GPU:
                tokens_tensor = tokens_tensor.to('cuda')
                maskALL_tensor = maskALL_tensor.to('cuda')
            hidden_states = self._model.extract_features(tokens_tensor)
            senMatrix = hidden_states * torch.unsqueeze(maskALL_tensor, 2)  # 使用掩码将多余词屏蔽
            senMatrix = torch.cat([senMatrix[:, 1:, :], senMatrix[:, :1, :]], dim=1)  # 移除 [CLS]
            senMatrix = np.array(senMatrix.cpu())
        return senMatrix, np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)  # np.array

    def _getSenVectorsModel_seg(self, senF_L, senB_L):
        senFrontMaxLen = self._senFrontMaxLen
        senBackMaxLen = self._senBackMaxLen
        senMaxLen = self._senMaxLen
        # 只训练前部
        self._senFrontMaxLen = senFrontMaxLen
        self._senBackMaxLen = 0
        self._senMaxLen = senFrontMaxLen
        senMatrix_f, sentences_mask_f_L, segPos_f_L, mid_f_L = self._getSenVectorsModel(senF_L, senB_L)  # 维度 3,2,2,1
        # 只训练后部
        self._senFrontMaxLen = 0
        self._senBackMaxLen = senBackMaxLen
        self._senMaxLen = senBackMaxLen
        senMatrix_b, sentences_mask_b_L, segPos_b_L, mid_b_L = self._getSenVectorsModel(senF_L, senB_L)
        # 复原
        self._senFrontMaxLen = senFrontMaxLen
        self._senBackMaxLen = senBackMaxLen
        self._senMaxLen = senMaxLen
        # 合并
        senMatrix, sentences_mask_L, segPos_L, mid_L = [], [], [], []
        for i in range(senMatrix_f.shape[0]):
            len_f = int(sum(sentences_mask_f_L[i]))
            len_b = int(sum(sentences_mask_b_L[i]))
            senMatrix.append(np.concatenate([senMatrix_f[i][:len_f], senMatrix_b[i][:len_b], np.zeros((senMaxLen-len_f-len_b, self._word_dim))], axis=0))
            sentences_mask_L.append([1.]*(len_f+len_b) + [0.]*(senMaxLen-len_f-len_b))
            segPos_L.append(segPos_f_L[i] + [j+len_f for j in segPos_b_L[i][1:]])  # segPos_b_L[i] 去掉0
            mid_L.append(len_f)
        return np.array(senMatrix), np.array(sentences_mask_L), np.array(segPos_L), np.array(mid_L)

    def _getAvgWordEmbedding(self):
        candSenVec, testSenVec = [], []

        senF_L = self._candTextF_sentence_L + self._testTextF_sentence_L  # [(候选集id,[文本前部]),..]
        senB_L = self._candTextB_sentence_L + self._testTextB_sentence_L
        if self._senID_mat_len_seg_mid_h5:
            if self._senID_no_D:
                useH5py = True
                print('从h5py文件中读取')
            else:
                useH5py = False
                print('将存储在h5py文件中')
                if self._segTrain:
                    print('将front和back分开训练...')
                else:
                    print('将front和back一起训练...')
        else:
            useH5py = False
        tokenNum_max_min_avg = [0, 10**10, 0]

        for i in tqdm(range(0, len(senF_L), self._bitchSize), '计算平均句向量'):
            F_L, B_L = [], []
            no_S = set()
            id_L = []
            for (id_f, sen_f), (id_b, sen_b) in zip(senF_L[i: i+self._bitchSize], senB_L[i: i+self._bitchSize]):
                assert id_f == id_b, '前部 后部 编号不对应!'
                if useH5py:
                    no_S.add(self._senID_no_D[id_f])
                id_L.append(id_f)
                F_L.append(sen_f)
                B_L.append(sen_b)
            if useH5py:
                no_L = sorted(no_S)
                no_L_L = [[no_L[0]]]
                for j in no_L[1:]:  # 合并连续的序号
                    if no_L_L[-1][-1] == j - 1:
                        no_L_L[-1].append(j)
                    else:
                        no_L_L.append([j])
                matrixs_L_L, lengths_L_L = [], []
                for j in no_L_L:  # 分批次读取
                    matrixs_L_L.append(self._senID_mat_len_seg_mid_h5['matrix'][j])
                    lengths_L_L.append(self._senID_mat_len_seg_mid_h5['length'][j])
                matrixs_L = np.concatenate(matrixs_L_L, axis=0)
                lengths_L = np.concatenate(lengths_L_L, axis=0)
                no_th_D = {j: k for k, j in enumerate(no_L)}
                for j in id_L:  # 获得句向量
                    m = matrixs_L[no_th_D[self._senID_no_D[j]]]
                    l = lengths_L[no_th_D[self._senID_no_D[j]]]
                    candSenVec.append(np.sum(m, axis=0) / l)
                    if l > tokenNum_max_min_avg[0]:
                        tokenNum_max_min_avg[0] = l
                    if l < tokenNum_max_min_avg[1]:
                        tokenNum_max_min_avg[1] = l
                    tokenNum_max_min_avg[2] += l
            else:
                if self._segTrain:
                    senMatrix, sentences_mask_L, segPos_L, mid_L = self._getSenVectorsModel_seg(F_L, B_L)
                else:
                    senMatrix, sentences_mask_L, segPos_L, mid_L = self._getSenVectorsModel(F_L, B_L)
                sentences_mask_L = np.sum(sentences_mask_L, axis=1)

                if self._senID_mat_len_seg_mid_h5:
                    senID = self._senID_mat_len_seg_mid_h5['senID']
                    matrix = self._senID_mat_len_seg_mid_h5['matrix']
                    length = self._senID_mat_len_seg_mid_h5['length']
                    segPos = self._senID_mat_len_seg_mid_h5['segPos']
                    mid = self._senID_mat_len_seg_mid_h5['mid']
                    # 增加大小
                    senID.resize([senID.shape[0] + len(id_L)])
                    matrix.resize([matrix.shape[0] + len(id_L), matrix.shape[1], matrix.shape[2]])
                    length.resize([length.shape[0] + len(id_L)])
                    segPos.resize([segPos.shape[0] + len(id_L)])
                    mid.resize([mid.shape[0] + len(id_L)])

                    senID[-len(id_L):] = id_L
                    matrix[-len(id_L):] = senMatrix
                    length[-len(id_L):] = sentences_mask_L
                    segPos[-len(id_L):] = segPos_L
                    mid[-len(id_L):] = mid_L
                    self._senID_mat_len_seg_mid_h5.flush()
                    for j in id_L:  # 如果有重复索引会导致一些冗余
                        self._senID_no_D[j] = len(self._senID_no_D)
                for j in range(len(id_L)):  # 获得句向量
                    l = sentences_mask_L[j]
                    candSenVec.append(np.sum(senMatrix[j], axis=0) / l)
                    if l > tokenNum_max_min_avg[0]:
                        tokenNum_max_min_avg[0] = l
                    if l < tokenNum_max_min_avg[1]:
                        tokenNum_max_min_avg[1] = l
                    tokenNum_max_min_avg[2] += l
        tokenNum_max_min_avg[2] /= len(senF_L)
        print('tokenNum_max_min_avg: %s' % str(tokenNum_max_min_avg))

        testSenVec = candSenVec[len(self._candTextF_sentence_L):]  # 和下面顺序不能反
        candSenVec = candSenVec[:len(self._candTextF_sentence_L)]
        return np.array(candSenVec), np.array(testSenVec)  # np.array

    @staticmethod
    def _getTextInfor(dataIRaddress):
        candTextF_sentence_L = []  # [(候选集id,[文本前部]),..]
        candTextB_sentence_L = []  # [(候选集id,[文本后部]),..]
        testTextF_sentence_L = []  # [(测试集id,[文本前部]),..]
        testTextB_sentence_L = []  # [(测试集id,[文本后部]),..]
        test_candidateS_D = {}  # {测试集id:候选集id set,..}

        with open(dataIRaddress, 'r', encoding='utf-8') as r:
            for i, line in tqdm(enumerate(r), '获取文本信息'):
                if i == 0:
                    test_candidateS_D = eval(line.strip())
                    continue
                line = line.split('\t')
                textID = line[0]
                if textID in test_candidateS_D:
                    testTextF_sentence_L.append((textID, 句子清洗(line[1]).split()))
                    testTextB_sentence_L.append((textID, 句子清洗(line[2]).split()))
                else:
                    candTextF_sentence_L.append((textID, 句子清洗(line[1]).split()))
                    candTextB_sentence_L.append((textID, 句子清洗(line[2]).split()))
        return candTextF_sentence_L, candTextB_sentence_L, testTextF_sentence_L, testTextB_sentence_L

    def computeIRcosSimMatrix(self):
        print('cos计算中...')
        # 每一行表示一个候选集文本和所有测试集文本的相似度
        cand_test_sim_L = (np.dot(self._testSenVec, self._candSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._candSenVec, axis=1), axis=0))).T

        # 还原编号
        test_cand8sim_D = {}  # {测试集文本编号:{候选集文本编号:相似度,..},..}
        for i in tqdm(range(len(cand_test_sim_L)), '还原编号, 候选集数'):
            for j, sim in enumerate(cand_test_sim_L[i]):
                testID = self._testTextF_sentence_L[j][0]
                candID = self._candTextF_sentence_L[i][0]
                if testID in test_cand8sim_D:
                    test_cand8sim_D[testID][candID] = sim
                else:
                    test_cand8sim_D[testID] = {candID: sim}
        # 排序
        for k in tqdm(test_cand8sim_D.keys(), '排序'):
            test_cand8sim_D[k] = sorted(test_cand8sim_D[k].items(), key=lambda t: t[1], reverse=True)
        self._test_cand8sim_D = test_cand8sim_D  # {测试集文本编号:[(候选集文本编号,相似度),..],..}
        test_cand_D = {k: [i[0] for i in v] for k, v in test_cand8sim_D.items()}
        return test_cand_D  # {测试集文本编号:[候选集文本编号,..]}

    def computeTCdisMatrix(self):
        '''
        首先数据集必须是分类数据集, 其次分类信息必须要在id双下划线分割的第二个位置
        :param 进程数:
        :return:
        '''
        print('dis计算中...')
        # 每一行表示一个候选集文本和所有测试集文本的距离(相似度的负数)
        test_cand_dis_L = -np.dot(self._testSenVec, self._candSenVec.T) / np.dot(np.expand_dims(np.linalg.norm(self._testSenVec, axis=1), axis=1), np.expand_dims(np.linalg.norm(self._candSenVec, axis=1), axis=0))

        # 获取标签
        candLabel_L = []
        testLabel_L = []
        for textID, _ in self._candTextF_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            candLabel_L.append(label)
        for textID, _ in self._testTextF_sentence_L:
            label = textID.split('__')[1]  # 标签在id中的位置
            testLabel_L.append(label)
        return test_cand_dis_L, candLabel_L, testLabel_L

    def __del__(self):
        if self._senID_mat_len_seg_mid_h5:
            self._senID_mat_len_seg_mid_h5.close()


if __name__ == '__main__':
    ap = 'data/CR dblp/'

    datasetPath = ap + 'dataset.text'
    senMatrixFolder = r'F:\data\_large_tem_file\CTE/' + ap
    # senMatrixFolder = ''
    modelName = DeepMethods.model.roberta_large
    if not senMatrixFolder or not os.path.exists(senMatrixFolder):
        senMatrixFolder = ap

    segTrain = False  # 是否将front和back分开获取词向量
    if segTrain:
        senMatrixPath = senMatrixFolder + 'bc_' + modelName + '_senID_mat_len_seg_mid-.h5'
    else:
        senMatrixPath = senMatrixFolder + 'bc_' + modelName + '_senID_mat_len_seg_mid.h5'

    startTime = time.time()
    DeepMethods_obj = DeepMethods(datasetPath=datasetPath,
                                  bitchSize=20,
                                  GPU=True,
                                  senFrontMaxLen=200,
                                  senBackMaxLen=200,
                                  modelName=modelName,
                                  # 如果存在则自动读取, .h5前加"-"表示分开文本训练的词向量
                                  senMatrixPath=senMatrixPath,
                                  model_path='data/-elmo-model/elmo_2x4096_512_2048cnn_2xhighway_5.5B_',  # 用于EMLo
                                  segTrain=segTrain,  # 是否将front和back分开获取词向量
                                  )

    # 检索数据集评估
    test_cand_D = DeepMethods_obj.computeIRcosSimMatrix()
    IR评估_obj = IR评估(标签地址=datasetPath)
    IR评估_obj.评估(预测标签=test_cand_D, topN=20, 简化=True, 输出地址='', 输出控制台=True)

    # 分类数据集评估
    # test_cand_dis_L, candLabel_L, testLabel_L = DeepMethods_obj.computeTCdisMatrix()
    # TC评估.距离矩阵评估(test_cand_dis_L, candLabel_L, testLabel_L,
    #             n_neighbors=[3, 6, 9, 12, 15, 18],
    #             n_jobs=8, knn使用距离加权=False, 输出控制台=True)

    print('%.2fm' % ((time.time() - startTime) / 60))
