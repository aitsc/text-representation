from tqdm import tqdm
import pickle
import torch
import numpy as np
from pprint import pprint
from allennlp.commands.elmo import ElmoEmbedder
from pytorch_transformers import *

a='target mass correction based on the re scaled parton densities?'.split()
a=tuple(a)

# print('BERT')
# with torch.no_grad():
#     pre='bert-large-uncased'
#     tokenizer = BertTokenizer.from_pretrained(pre)
#     c=tokenizer.encode('[CLS] ' + ' '.join(a) + ' [SEP]')
#     print(c)
#     print(tokenizer.convert_ids_to_tokens(c))
#     model = BertModel.from_pretrained(pre)
#     model.eval()
#     hidden_states = model(torch.tensor([c+[0]]), attention_mask=torch.tensor([[1]*14+[0]*1]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c+[0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c]))[0]
#     print(hidden_states)

print('XLNet')
with torch.no_grad():
    pre='xlnet-large-cased'
    tokenizer = XLNetTokenizer.from_pretrained(pre)
    c=tokenizer.encode(' '.join(a))
    length = len(c)
    print(c)
    print(tokenizer.convert_ids_to_tokens(c))
    model = XLNetModel.from_pretrained(pre)
    model.eval()
    mask = torch.tensor([[1.]*length+[0.]*1])
    hidden_states = model(torch.tensor([c+[0]]), attention_mask=mask)[0]
    print(hidden_states * torch.unsqueeze(mask, 2))
    hidden_states = model(torch.tensor([c+[0]]))[0]
    print(hidden_states)
    hidden_states = model(torch.tensor([c]))[0]
    print(hidden_states)
    print(hidden_states.shape)

# print('GPT2')  # 没有掩码, 充0相等
# with torch.no_grad():
#     pre='gpt2-medium'
#     tokenizer = GPT2Tokenizer.from_pretrained(pre)
#     c=tokenizer.encode(' '.join(a))
#     print(c)
#     print(tokenizer.convert_ids_to_tokens(c))
#     model = GPT2Model.from_pretrained(pre)
#     model.eval()
#     hidden_states = model(torch.tensor([c+[0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c]))[0]
#     print(hidden_states)
#     print(hidden_states.shape)

# print('GPT')  # 没有掩码, 充0有时不相等, 不是 causal model 吗?
# with torch.no_grad():
#     pre='openai-gpt'
#     tokenizer = OpenAIGPTTokenizer.from_pretrained(pre)
#     c=tokenizer.encode(' '.join(a))
#     print(c)
#     print(tokenizer.convert_ids_to_tokens(c))
#     model = OpenAIGPTModel.from_pretrained(pre)
#     model.eval()
#     hidden_states = model(torch.tensor([c+[0, 0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c+[0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c]))[0]
#     print(hidden_states)
#     print(hidden_states.shape)

# print('TransfoXLModel')  # 没有掩码, 充0相等
# with torch.no_grad():
#     pre='transfo-xl-wt103'
#     tokenizer = TransfoXLTokenizer.from_pretrained(pre)
#     c=tokenizer.encode(' '.join(a))
#     length = len(c)
#     print(c)
#     print(tokenizer.convert_ids_to_tokens(c))
#     model = TransfoXLModel.from_pretrained(pre)
#     model.eval()
#     hidden_states = model(torch.tensor([c+[0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c]))[0]
#     print(hidden_states)
#     print(hidden_states.shape)

# print('XLMModel')
# with torch.no_grad():
#     pre='xlm-mlm-en-2048'
#     tokenizer = XLMTokenizer.from_pretrained(pre)
#     c=tokenizer.encode(' '.join(a))
#     length = len(c)
#     print(c)
#     print(tokenizer.convert_ids_to_tokens(c))
#     model = XLMModel.from_pretrained(pre)
#     model.eval()
#     mask = torch.tensor([[1.]*length+[0.]*1])
#     hidden_states = model(torch.tensor([c+[0]]), attention_mask=mask)[0]
#     print(hidden_states * torch.unsqueeze(mask, 2))
#     hidden_states = model(torch.tensor([c+[0]]))[0]
#     print(hidden_states)
#     hidden_states = model(torch.tensor([c]))[0]
#     print(hidden_states)
#     print(hidden_states.shape)

# print('ELMo')  # 掩码部分向量自动充0
# pretrained_model = r'D:\data\code\python\paper\text representation\data\-elmo-model\elmo_2x4096_512_2048cnn_2xhighway_5.5B_'
# model = ElmoEmbedder(pretrained_model + 'options.json', pretrained_model + 'weights.hdf5', cuda_device=-1)
# elmo_embedding, _ = model.batch_to_embeddings([list(a), list(a)])
# # 句向量 = np.sum(np.array(elmo_embedding), axis=1) / elmo_embedding.shape[1]
# # print(句向量)
# print(np.array(elmo_embedding).shape)
# elmo_embedding, _ = model.batch_to_embeddings([list(a), list(a), list(a)[:7], list(a)[:4]+['it'], list(a), list(a)])
# # 句向量 = np.sum(np.array(elmo_embedding), axis=1) / elmo_embedding.shape[1]
# # print(句向量)
# print(np.array(elmo_embedding).shape)
# print(np.concatenate([i.squeeze() for i in elmo_embedding.chunk(3,1)],axis=2).shape)