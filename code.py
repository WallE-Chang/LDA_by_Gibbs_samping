#!/usr/bin/env python
# coding=utf-8
'''
@Author: Wanli Chang
@Date: 2020-01-11 14:47:25
@LastEditTime : 2020-01-13 17:50:18
@LastEditors  : Wanli Chang
'''
#!/usr/bin/env python
# coding=utf-8
'''
@Author: Wanli Chang
@Date: 2020-01-11 14:47:25
@LastEditTime : 2020-01-13 17:21:05
@LastEditors  : Wanli Chang
'''
# %%
import time
import numpy as np
import codecs
import jieba
import re
import random
from typing import Tuple, Dict, List
from tqdm import tqdm

#%%
def train_preprocessing(stopwords_file_path: str, train_data: str) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
    """
    预处理(分词，去停用词，为每个word赋予一个编号，文档使用word编号的列表表示)

    Arguments:
        stopwords_file_path {str} -- [停用词文件存放路径]
        doc_file_path {str} -- [用户训练的文本文件]

    Returns:
        Tuple[List[List[int]],Dict[str,int],Dict[int,str]] -- [预处理后的 docs, word2id, id2word]
    """

    # 读取停止词文件
    with codecs.open(stopwords_file_path, 'r', 'utf-8') as file:
        stopwords = [line.strip() for line in file]

    word2id = {}
    id2word = {}
    docs = []
    current_document = []
    current_word_id = 0

    for document in tqdm(train_data,bar_format='train_preprocessing'):
        # 分词
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            # 单词长度大于1并且不包含数字并且不是停止词
            if len(word) > 1 and not re.search('[0-9]', word) and word not in stopwords:
                if word in word2id:
                    current_document.append(word2id[word])
                else:
                    current_document.append(current_word_id)
                    word2id[word] = current_word_id
                    id2word[current_word_id] = word
                    current_word_id += 1
        docs.append(current_document)
        current_document = []
    return docs, word2id, id2word

def test_preprocessing(test_data,word2idx):
    docs = []
    current_document = []
    for document in tqdm(test_data,bar_format='test_preprocessing'):
        # 分词
        segList = jieba.cut(document)
        for word in segList:
            word = word.lower().strip()
            word_idx = word2idx.get(word)
            if word_idx is not None:
                current_document.append(word_idx)
        docs.append(current_document)
        current_document=[]
    return docs

def get_tarin_test_data(docs_path:str,test_size:float=0.2,shuffle=True):
    with codecs.open(docs_path, 'r', 'utf-8') as file:
        documents = [document.strip() for document in file]
    if shuffle:
        random.shuffle(documents)
    test_data = documents[:int(len(documents)*test_size)]
    train_data = documents[int(len(documents)*test_size):]
    return train_data,test_data
    


def random_initialize(docs: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array) -> Tuple[List[List[int]], np.array, np.array, np.array]:
    """
    初始化，为文档中的每个词采样 topic:z_m_n = k ~ Mult(1/k), 并更新 all count variables

    Arguments:
        docs {List[List[int]]} -- [预处理后的 docs]
        n_doc_topic {np.array} -- [shape:(M,K)]
        n_topic_term {np.array} -- [shape(K,V)]
        n_topic_term_sum {np.array} -- [shape:(K,)]

    Returns:
        Tuple[List[List[int]],np.array,np.array,np.array] -- [文档中每个词的主题，更新后的n_doc_topic，更新后的n_topic_term, 更新后的n_topic_term_sum]
    """
    Z = []  # 存储文档中每个词的主题
    for m, doc in enumerate(docs):
        zCurrentDoc = []
        for n, word in enumerate(doc):
            z_m_n = random.randint(a=0, b=K-1)  # z_m_n = k ~ Mult(1/k)
            zCurrentDoc.append(z_m_n)
            n_doc_topic[m, z_m_n] += 1
            n_topic_term[z_m_n, word] += 1
            n_topic_term_sum[z_m_n] += 1
        Z.append(zCurrentDoc)
    return Z, n_doc_topic, n_topic_term, n_topic_term_sum


def gibbs_sampling(docs: List[List[int]], Z: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array, alpha: float, beta: float) -> Tuple[List[List[int]], np.array, np.array, np.array]:
    """
    为文档中的每个词重新采样 topic，并更新 all count variables

    Arguments:
        docs {List[List[int]]} -- [预处理后的 docs]
        Z {List[List[int]]} -- [文档中每个词的主题]
        n_doc_topic {np.array} -- [shape:(M,K)]
        n_topic_term {np.array} -- [shape(K,V)]
        n_topic_term_sum {np.array} -- [shape:(K,)]
        alpha {float} -- [ doc 狄利克雷分布的超参数]]
        beta {float} -- [ word 狄利克雷分布的超参数]

    Returns:
        Tuple[List[List[int]], np.array, np.array, np.array] -- [采样后文档中每个词的主题，更新后的n_doc_topic，更新后的n_topic_term, 更新后的n_topic_term_sum]
    """
    V = n_topic_term.shape[1]
    for m, doc in enumerate(docs):
        for n, t in enumerate(doc):
            k = Z[m][n]  # 当前 word 的主题 k
            # 将当前文档当前单词原topic相关计数减去1
            n_doc_topic[m, k] -= 1
            n_topic_term[k, t] -= 1
            n_topic_term_sum[k] -= 1
            # 重新计算当前文档当前单词属于每个topic的概率
            part_13 = n_topic_term[:, t]+beta  # K vector
            part_15 = n_doc_topic[m, ]+alpha  # K vector
            part_14 = n_topic_term_sum + V * beta  # K vector
            p_z = np.divide(np.multiply(part_13, part_15),
                            part_14)  # formula (15)
            # 按照计算出的分布进行采样
            k = np.random.multinomial(1, p_z / p_z.sum()).argmax()
            Z[m][n] = k
            # 将当前文档当前单词新采样的topic相关计数加上1
            n_doc_topic[m, k] += 1
            n_topic_term[k, t] += 1
            n_topic_term_sum[k] += 1
    return Z, n_doc_topic, n_topic_term, n_topic_term_sum


def calculate_perplexity(test_docs: List[List[int]], theta: np.array, phi: np.array) -> float:
    """
    计算 test_docs 的 perplexity，也就是计算在 test_docs 里面所有词出现的概率，对主题 z 求积分

    paper: http://jmlr.org/papers/volume3/blei03a/blei03a.pdf  7.1 Document modeling

    Arguments:
        test_docs {List[List[int]]} -- [预处理后的 test_docs]
        theta {np.array} -- [shape:(M,K)]
        phi {np.array} -- [shape(K,V)]

    Returns:
        float -- [perplexity]
    """
    # 计算在 test_docs 里面所有词出现的概率，对主题 z 求积分
    sum_n_d = 0
    sum_log_p = 0.0

    probability_doc_topic =theta
    probability_topic_term = phi

    for m, doc in enumerate(test_docs):
        for n, word in enumerate(doc):
            sum_log_p = sum_log_p + \
                np.log(
                    np.dot(probability_topic_term[:, word], probability_doc_topic[m, :]))
            sum_n_d = sum_n_d + 1
    return np.exp(sum_log_p/(-sum_n_d))

def calculate_theta(n_doc_topic:np.array,alpha:float)->np.array:
    """
    计算参数 theta, 也就是把 n_doc_topic+alpha 行归一化
    
    Arguments:
        n_doc_topic {np.array} -- [shape:(M,K)]
        alpha {float} -- [ doc 狄利克雷分布的超参数]]
    
    Returns:
        np.array -- [shape:(M,K)]
    """
    return (n_doc_topic+alpha)/(n_doc_topic+alpha).sum(1).reshape(-1, 1)

def calculate_phi(n_topic_term:np.array,beta:float)->np.array:
    """
    计算参数 phi, 也就是把 n_topic_term+beta 行归一化
    
    Arguments:
        n_topic_term {np.array} -- [shape(K,V)]
        beta {float} -- [ word 狄利克雷分布的超参数]
    
    Returns:
        np.array -- [shape(K,V)]
    """
    return (n_topic_term+beta)/(n_topic_term+beta).sum(1).reshape(-1, 1)

def calculate_error(n_doc_topic_pre: np.array, n_topic_term_pre: np.array, n_doc_topic: np.array, n_topic_term: np.array) -> float:
    """
    计算上一轮迭代和这一轮迭代 n_doc_topic 和 n_topic_term 的变化度，以衡量是否收敛

    Arguments:
        n_doc_topic_pre {np.array} -- [shape:(M,K)]
        n_topic_term_pre {np.array} -- [shape(K,V)]
        n_doc_topic {np.array} -- [shape:(M,K)]
        n_topic_term {np.array} -- [shape(K,V)]

    Returns:
        float -- [变化度]
    """
    error_doc_topic = np.sum(
        np.abs(n_doc_topic-n_doc_topic_pre))/n_doc_topic_pre.sum()
    error_topic_term = np.sum(
        np.abs(n_topic_term-n_topic_term_pre))/n_topic_term_pre.sum()
    error = np.mean((error_doc_topic, error_topic_term))
    return error


def get_top_topic_word(n_topic_term: np.array, id2word: Dict[int, str], max_topic_word_num: int = 10) -> List[List[str]]:
    """
    得到每个主题出现概率最高的 max_topic_word_num 个单词

    Arguments:
        n_topic_term {np.array} -- [shape(K,V)]
        id2word {Dict[int,str]} -- [index to word]

    Keyword Arguments:
        max_topic_word_num {int} -- [每个主题只显示的单词数量] (default: {10})

    Returns:
        List[List[str]] -- [每个主题出现概率最高的 max_topic_word_num 个单词]
    """
    topic_words = []
    for z in range(0, K):
        ids = n_topic_term[z, :].argsort()[::-1]
        topic_word = []
        for word_num, word_index in enumerate(ids):
            if word_num < max_topic_word_num:
                topic_word.append(id2word[word_index])
            else:
                break
        topic_words.append(topic_word)
    return topic_words

# %%
if __name__ == "__main__":
    import seaborn as sns
    import pandas as pd
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.set(style="darkgrid")
    
    # file path
    stopwords_file_path = 'stopwords.dic'
    doc_file_path = 'dataset_cn.txt'

    # super parameters
    alpha = 5  # doc 狄利克雷分布的超参数
    beta = 0.1  # word 狄利克雷分布的超参数
    iterationNum = 50  # 迭代次数
    K = 10  # 主题个数

    Z = []  # 存储文档中每个词的主题,len(Z)=M and len(Z[m])= N_m
# %% load data
    train_data,test_data = get_tarin_test_data(doc_file_path)
# %% 数据预处理
    train_docs, word2id, id2word = train_preprocessing(stopwords_file_path, train_data)
    test_docs = test_preprocessing(test_data,word2id)
#%%
    M = len(train_docs)  # 有多少个文档
    V = len(word2id)  # 词袋的里面有多少个词

    # zero all count variables
    n_doc_topic = np.zeros([M, K])
    n_topic_term = np.zeros([K, V])
    n_topic_term_sum = np.zeros([K])

    # get topic by random sampling
    Z, n_doc_topic, n_topic_term, n_topic_term_sum = random_initialize(
        train_docs, n_doc_topic, n_topic_term, n_topic_term_sum)
    
    # %%
    # Gibbs sampling
    error_list=[]
    for i in range(0, iterationNum):
        n_doc_topic_pre, n_topic_term_pre = n_doc_topic.copy(), n_topic_term.copy()
        Z, n_doc_topic, n_topic_term, n_topic_term_sum = gibbs_sampling(
            train_docs, Z, n_doc_topic, n_topic_term, n_topic_term_sum, alpha, beta)
        error = calculate_error(
            n_doc_topic_pre, n_topic_term_pre, n_doc_topic, n_topic_term)
        print(time.strftime('%X'),f'error of iteration {i}: {error}')
        error_list.append(error)

    # 得到每个主题出现概率最高的 max_topic_word_num 个单词
    topic_words = get_top_topic_word(n_topic_term, id2word)

    # %%
    # visualization
    history = pd.DataFrame({"error":error_list}).reset_index()
    sns.lineplot(x="index", y="error",
                data=history)
    # %%
    theta = calculate_theta(n_doc_topic,alpha)
    phi=calculate_phi(n_topic_term,beta)

    #计算训练集和测试集的 perplexity
    train_perplexity = calculate_perplexity(train_docs,theta,phi)
    test_perplexity = calculate_perplexity(test_docs,theta,phi)
    print(f'perplexity of train/test: {train_perplexity}/{test_perplexity}')




# %%
