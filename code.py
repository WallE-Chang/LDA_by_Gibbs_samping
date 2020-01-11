#!/usr/bin/env python
# coding=utf-8
'''
@Author: Wanli Chang
@Date: 2020-01-11 14:47:25
@LastEditTime : 2020-01-11 17:27:55
@LastEditors  : Wanli Chang
'''
import numpy as np
import time
import codecs
import jieba
import re

# 预处理(分词，去停用词，为每个word赋予一个编号，文档使用word编号的列表表示)
def preprocessing():
    # 读取停止词文件
    file = codecs.open('stopwords.dic','r','utf-8')
    stopwords = [line.strip() for line in file] 
    file.close()
    
    # 读数据集
    file = codecs.open('dataset.txt','r','utf-8')
    documents = [document.strip() for document in file] 
    file.close()
    
    word2id = {}
    id2word = {}
    docs = []
    currentDocument = []
    currentWordId = 0
    
    for document in documents:
        # 分词
        segList = jieba.cut(document)
        for word in segList: 
            word = word.lower().strip()
            # 单词长度大于1并且不包含数字并且不是停止词
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    currentDocument.append(word2id[word])
                else:
                    currentDocument.append(currentWordId)
                    word2id[word] = currentWordId
                    id2word[currentWordId] = word
                    currentWordId += 1
        docs.append(currentDocument)
        currentDocument = []
    return docs, word2id, id2word

if __name__ == "__main__":
    alpha = 5
    beta = 0.1	
    iterationNum = 50
    Z = []
    K = 10  #主题个数
    docs, word2id, id2word = preprocessing()
    M = len(docs) # 有多少个文档
    V = len(word2id) # 词袋的里面有多少个词
    ndz = np.zeros([M, K]) + alpha #
    nzw = np.zeros([K, M]) + beta
    nz = np.zeros([K]) + M * beta