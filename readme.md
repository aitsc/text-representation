# Introduction
This code is the implementation of the paper "Coherence-Based Distributed Document Representation Learning for Scientific Documents".

If you find this code useful, please cite the following paper:
```
@article{tan2022coherence,
  title = {Coherence-Based Distributed Document Representation Learning for Scientific Documents},
  author = {Tan, Shicheng and Zhao, Shu and Zhang, Yanping},
  journal = {arXiv},
  year = {2022},
  type = {Journal Article}
}
```

# Run
1. Installation environment (ref. requirements.txt)
2. Download data: Link: https://pan.baidu.com/s/1EEJk0_P55Ov5ReXsmyVZPA Password: rkh0
3. python _av_CTE.py

# 信息检索数据运行指南

1. 数据处理（4个文件）：使用“...data helper-IR.py”获取3份数据，原始数据处理暂存文件、原始数据处理暂存文件的语料、构建的数据集，然后使用“_aj_get dataset corpus.py”获得 构建的数据集的语料
2. 词向量训练（4个文件）：使用“_ak_get word embedding.py”训练第一步的2个语料得到2个词表和2个词向量文件，glove需要去除后缀名“.txt”
3. 运行5次“_al_em-avg.py”得到5个结果，avg-word2vec、avg-word2vec(globe)、avg-glove、avg-glove(globe)、random embedding
4. 运行“_ac_tf-idf.py”得到一个距离矩阵和1个结果，矩阵用于CTE方法
5. LDA、doc2vec、BM25、LSI、GPT2、XLNet、GPT、Transformer-XL、XLM 对应文件各运行一次得到9个结果
6. 运行“_ah_WMD.py”4次得到4个结果，WMD-word2vec、WMD-word2vec(globe)、WMD-glove、WMD-glove(globe)
7. 运行“_at_BERT.py”2次得到2个结果，BERT-Large uncased、BERT-Large uncased(wwm)
8. 运行“_at_ELMo.py”2次得到2个结果，ELMo-Original(5.5B)、ELMo-Original(5.5B,级联)
9. 运行“_av_CET.py”13次得到13个结果，基于 random embedding 等13种基础词向量
