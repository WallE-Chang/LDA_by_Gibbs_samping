import time
import numpy as np
import codecs
import jieba
import re
import random
from typing import Tuple, Dict, List
from tqdm import tqdm


class lda():
    def __init__(self, alpha: float = 5.0, beta: float = 0.1, K: int = 10, stopwords_file_path='stopwords.dic'):
        """
        获得超参数

        Keyword Arguments:
            alpha {float} -- [doc 狄利克雷分布的超参数] (default: {5.0})
            beta {float} -- [word 狄利克雷分布的超参数] (default: {0.1})
            K {int} -- [主题个数] (default: {10})
        """

        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.stopwords_file_path = stopwords_file_path
        with codecs.open(self.stopwords_file_path, 'r', 'utf-8') as file:
            self.stopwords = [line.strip() for line in file]
        
    def fit(self,train_data,iterationNum=50):
        self.train_docs, self.word2id, self.id2word = self.train_preprocessing(train_data)
        M = len(self.train_docs)  # 有多少个文档
        V = len(self.word2id)  # 词袋的里面有多少个词

        # zero all count variables
        self.n_doc_topic = np.zeros([M, self.K])
        self.n_topic_term = np.zeros([self.K, V])
        self.n_topic_term_sum = np.zeros([self.K])

        # get topic by random sampling
        self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum = self._random_initialize(
            self.train_docs, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum)

        # Gibbs sampling
        error_list=[]
        for i in range(0, iterationNum):
            n_doc_topic_pre, n_topic_term_pre = self.n_doc_topic.copy(), self.n_topic_term.copy()
            self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum = self._gibbs_sampling(
                self.train_docs, self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum, self.alpha, self.beta)
            error = self.calculate_error(
                n_doc_topic_pre, n_topic_term_pre, self.n_doc_topic, self.n_topic_term)
            print(time.strftime('%X'),f'error of iteration {i}: {error}')
            error_list.append(error)
        # TODO:


    def data_preprocessing(self, train_data, test_data):
        self.train_docs, self.word2id, self.id2word = self.train_preprocessing(train_data)
        self.test_docs = self.test_preprocessing(test_data)


    def train_preprocessing(self, train_data: List[List[str]]) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        """
        预处理(分词，去停用词，为每个word赋予一个编号，文档使用word编号的列表表示)

        Arguments:
            stopwords_file_path {str} -- [停用词文件存放路径]
            doc_file_path {str} -- [用户训练的文本文件]

        Returns:
            Tuple[List[List[int]],Dict[str,int],Dict[int,str]] -- [预处理后的 docs, word2id, id2word]
        """

        # 读取停止词文件

        word2id = {}
        id2word = {}
        docs = []
        current_document = []
        current_word_id = 0

        for document in tqdm(train_data, bar_format='train_preprocessing'):
            # 分词
            segList = jieba.cut(document)
            for word in segList:
                word = word.lower().strip()
                # 单词长度大于1并且不包含数字并且不是停止词
                if len(word) > 1 and not re.search('[0-9]', word) and word not in self.stopwords:
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

    def test_preprocessing(self,test_data):
        docs = []
        current_document = []
        for document in tqdm(test_data, bar_format='test_preprocessing'):
            # 分词
            segList = jieba.cut(document)
            for word in segList:
                word = word.lower().strip()
                word_idx = self.word2id.get(word)
                if word_idx is not None:
                    current_document.append(word_idx)
            docs.append(current_document)
            current_document = []
        return docs

    def _random_initialize(self, docs: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array) -> Tuple[List[List[int]], np.array, np.array, np.array]:
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
                z_m_n = random.randint(a=0, b=self.K-1)  # z_m_n = k ~ Mult(1/k)
                zCurrentDoc.append(z_m_n)
                n_doc_topic[m, z_m_n] += 1
                n_topic_term[z_m_n, word] += 1
                n_topic_term_sum[z_m_n] += 1
            Z.append(zCurrentDoc)
        return Z, n_doc_topic, n_topic_term, n_topic_term_sum

    def _gibbs_sampling(self, docs: List[List[int]], Z: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array, alpha: float, beta: float) -> Tuple[List[List[int]], np.array, np.array, np.array]:
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

    staticmethod
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
