#!/usr/bin/env python
# coding=utf-8
'''
@Author: Wanli Chang
@Date: 2020-01-14 14:13:16
@LastEditTime : 2020-01-14 18:13:03
@LastEditors  : Wanli Chang
'''
# %%
import codecs
import random
from model.lda_model import lda_model


def get_tarin_test_data(docs_path: str, test_size: float = 0.2, shuffle=True):
    with codecs.open(docs_path, 'r', 'utf-8') as file:
        documents = [document for document in file]
    if shuffle:
        random.shuffle(documents)
    test_data = documents[:int(len(documents)*test_size)]
    train_data = documents[int(len(documents)*test_size):]
    return train_data, test_data

# %%
# file path
stopwords_file_path = 'stopwords.dic'
doc_file_path = 'dataset_cn.txt'

# super parameters
alpha = 5  # doc 狄利克雷分布的超参数
beta = 0.1  # word 狄利克雷分布的超参数
K = 10  # 主题个数

# load data and train test split
train_data, test_data = get_tarin_test_data(doc_file_path)
#%%
# init model
lda = lda_model(alpha=alpha,beta=beta,K=K,stopwords_file_path=stopwords_file_path)
lda.fit(train_data,iterationNum=50)


# %%
phi_test=lda.predict(test_data)
print(test_data[0])
print(lda.get_top_topic_word()[phi_test[0].argmax()])
# %%
print(f'perplexity of train/test:{lda.train_perplexity}/{lda.test_perplexity}')

