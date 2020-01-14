import codecs
import os
import random
import re
import time
from typing import Dict, List, Tuple

import jieba
import numpy as np
from tqdm import tqdm


class lda_model():
    def __init__(self, alpha: float = 5.0, beta: float = 0.1, K: int = 10, stopwords_file_path='stopwords.dic'):
        """
        获得超参数

        Keyword Arguments:
            alpha {float} -- [doc 狄利克雷分布的超参数] (default: {5.0})
            beta {float} -- [word 狄利克雷分布的超参数] (default: {0.1})
            K {int} -- [主题个数] (default: {10})
            stopwords_file_path {str} -- [停用词存放路径] (default: {'stopwords.dic'})
        """

        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.stopwords_file_path = stopwords_file_path
        if os.path.exists(self.stopwords_file_path):
            with codecs.open(self.stopwords_file_path, 'r', 'utf-8') as file:
                self.stopwords = [line.strip() for line in file]
        else:
            self.stopwords = []

    def fit(self, train_data: List[str], iterationNum: int = 50, visualize: bool = True):
        """
        lda 模型训练阶段

        Arguments:
            train_data {List[str]} -- [用户用于训练的文本,List 中的每一个元素是一篇 doc ]

        Keyword Arguments:
            iterationNum {int} -- [迭代次数] (default: {50})
            visualize {bool} -- [是否画出 error 曲线] (default: {True})
        """

        self.train_docs, self.word2id, self.id2word = self.train_preprocessing(
            train_data, stopwords=self.stopwords)
        M = len(self.train_docs)  # 有多少个文档
        V = len(self.word2id)  # 词袋的里面有多少个词

        # zero all count variables
        self.n_doc_topic = np.zeros([M, self.K])
        self.n_topic_term = np.zeros([self.K, V])
        self.n_topic_term_sum = np.zeros([self.K])

        # get topic by random sampling
        self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum = self._random_initialize_for_train(
            self.train_docs, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum)

        # Gibbs sampling
        error_list = []
        for i in range(0, iterationNum):
            n_doc_topic_pre, n_topic_term_pre = self.n_doc_topic.copy(), self.n_topic_term.copy()
            self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum = self._gibbs_sampling_for_train(
                self.train_docs, self.Z, self.n_doc_topic, self.n_topic_term, self.n_topic_term_sum, self.alpha, self.beta,)
            error = self._calculate_error_for_train(
                n_doc_topic_pre, n_topic_term_pre, self.n_doc_topic, self.n_topic_term)
            print(time.strftime('%X'), f'error of iteration {i}: {error}')
            error_list.append(error)

        # visualize error
        if visualize:
            self._visualize_error(error_list)

        # calculate theta and phi
        self.theta = self._calculate_theta(self.n_doc_topic, self.alpha)
        self.phi = self._calculate_phi(self.n_topic_term, self.beta)

        # calculate perplexity
        self.train_perplexity = self._calculate_perplexity(
            self.train_docs, self.theta, self.phi)

    def predict(self, test_data: List[str], iterationNum: int = 10, visualize: bool = True)->np.array:
        """
        lda 模型预测阶段
        
        Args:
            test_data (List[str]): [用户用于预测的文本,List 中的每一个元素是一篇 doc]
            iterationNum (int, optional): [迭代次数]. Defaults to 10.
            visualize (bool, optional): [是否画出 error 曲线]. Defaults to True.
        
        Returns:
            np.array: [预测文档的主题分布,shape:[M_test,K]]
        """
        test_docs = self.test_preprocessing(test_data, self.word2id)
        M_test = len(test_docs)  # test_data有多少个文档
        n_doc_topic_test = np.zeros([M_test, self.K])

        # get topic by random sampling
        Z_test, n_doc_topic_test = self._random_initialize_for_inference(
            test_docs, n_doc_topic_test)

        # Gibbs sampling
        error_list_test = []
        for i in range(0, iterationNum):
            n_doc_topic_test_pre = n_doc_topic_test.copy()
            Z_test, n_doc_topic_test = self._gibbs_sampling_for_inference(
                test_docs, Z_test, n_doc_topic_test, self.phi, self.alpha)
            error_test = self._calculate_error_for_inference(
                n_doc_topic_test_pre, n_doc_topic_test)
            print(time.strftime('%X'), f'error of iteration {i}: {error_test}')
            error_list_test.append(error_test)

        # visualize error
        if visualize:
            self._visualize_error(error_list_test)

        # calculate theta
        test_theta = self._calculate_theta(n_doc_topic_test, self.alpha)

        # calculate perplexity
        self.test_perplexity = self._calculate_perplexity(
            test_docs, test_theta, self.phi)

        return test_theta

    # ---------------------------------------------------------------------------- #
    #                              function for train                              #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def train_preprocessing(train_data: List[str], stopwords: List[str] = []) -> Tuple[List[List[int]], Dict[str, int], Dict[int, str]]:
        """
        预处理(分词，去停用词，为每个word赋予一个编号，文档使用word编号的列表表示)

        Arguments:
            train_data {List[str]} -- [用户训练的文本,List 中的每一个元素是一篇 doc ]

        Keyword Arguments:
            stopwords {List[str]} -- [停用词，List 中每一个元素是一个停用词] (default: {[]})

        Returns:
            Tuple[List[List[int]], Dict[str, int], Dict[int, str]] -- [预处理后的 docs, word2id, id2word]
        """
        word2id = {}
        id2word = {}
        docs = []
        current_document = []
        current_word_id = 0

        # 根据空格来分词
        train_data = [document.strip() for document in train_data]

        for document in tqdm(train_data, bar_format='train_preprocessing'):
            # jieba 分词
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

    @staticmethod
    def _random_initialize_for_train(docs: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array) -> Tuple[List[List[int]], np.array, np.array, np.array]:
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
        K = n_doc_topic.shape[1]  # 主题个数
        for m, doc in enumerate(docs):
            zCurrentDoc = []
            for n, word in enumerate(doc):
                z_m_n = random.randint(a=0, b=K-1)  # z_m_n = k ~ Mult(1/k)
                zCurrentDoc.append(z_m_n)
                # increment count variables
                n_doc_topic[m, z_m_n] += 1
                n_topic_term[z_m_n, word] += 1
                n_topic_term_sum[z_m_n] += 1
            Z.append(zCurrentDoc)
        return Z, n_doc_topic, n_topic_term, n_topic_term_sum

    @staticmethod
    def _gibbs_sampling_for_train(docs: List[List[int]], Z: List[List[int]], n_doc_topic: np.array, n_topic_term: np.array, n_topic_term_sum: np.array, alpha: float, beta: float) -> Tuple[List[List[int]], np.array, np.array, np.array]:
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

                doc_topic_probability = part_15
                topic_term_probability = np.divide(part_13, part_14)

                p_z = np.multiply(doc_topic_probability,
                                  topic_term_probability)
                # 按照计算出的分布进行采样
                k = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                Z[m][n] = k
                # 将当前文档当前单词新采样的topic相关计数加上1
                n_doc_topic[m, k] += 1
                n_topic_term[k, t] += 1
                n_topic_term_sum[k] += 1
        return Z, n_doc_topic, n_topic_term, n_topic_term_sum

    @staticmethod
    def _calculate_error_for_train(n_doc_topic_pre: np.array, n_topic_term_pre: np.array, n_doc_topic: np.array, n_topic_term: np.array) -> float:
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

    def get_top_topic_word(self, max_topic_word_num: int = 10) -> List[List[str]]:
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
        for z in range(0, self.K):
            ids = self.n_topic_term[z, :].argsort()[::-1]
            topic_word = []
            for word_num, word_index in enumerate(ids):
                if word_num < max_topic_word_num:
                    topic_word.append(self.id2word[word_index])
                else:
                    break
            topic_words.append(topic_word)
        return topic_words

    # ---------------------------------------------------------------------------- #
    #                               function for test                              #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def test_preprocessing(test_data: List[str], word2id: Dict[str, int]) -> List[List[int]]:
        """
        预处理(分词，去不在 word2id 中的词，并将 word 转化为 index)

        Arguments:
            test_data {List[str]} -- [用户预测的文本,List 中的每一个元素是一篇 doc]
            word2id {Dict[str,int]} -- [训练阶段产生的 word2id]]

        Returns:
            List[List[str]] -- [预处理后的 docs]
        """
        docs = []
        current_document = []

        # 根据空格来分词
        test_data = [document.strip() for document in test_data]

        for document in tqdm(test_data, bar_format='test_preprocessing'):
            # jieba 分词
            segList = jieba.cut(document)
            for word in segList:
                word = word.lower().strip()
                word_idx = word2id.get(word)
                if word_idx is not None:
                    current_document.append(word_idx)
            docs.append(current_document)
            current_document = []
        return docs

    @staticmethod
    def _random_initialize_for_inference(docs: List[List[int]], n_doc_topic: np.array) -> Tuple[List[List[int]], np.array]:
        """
        初始化，为文档中的每个词采样 topic:z_m_n = k ~ Mult(1/k), 只更新 n_doc_topic
        NOTE: 在 inference 过程中，phi 是固定的，也就是说 n_topic_term，n_topic_term_sum 不发生改变

        Arguments:
            docs {List[List[int]]} -- [预处理后的 docs]
            n_doc_topic {np.array} -- [shape:(M,K)]


        Returns:
            Tuple[List[List[int]],np.array] -- [文档中每个词的主题，更新后的n_doc_topic]
        """
        Z = []  # 存储文档中每个词的主题
        K = n_doc_topic.shape[1]  # 主题个数
        for m, doc in enumerate(docs):
            zCurrentDoc = []
            for n, word in enumerate(doc):
                z_m_n = random.randint(a=0, b=K-1)  # z_m_n = k ~ Mult(1/k)
                zCurrentDoc.append(z_m_n)
                n_doc_topic[m, z_m_n] += 1
            Z.append(zCurrentDoc)
        return Z, n_doc_topic

    @staticmethod
    def _gibbs_sampling_for_inference(docs: List[List[int]], Z: List[List[int]], n_doc_topic: np.array, phi: np.array, alpha: float) -> Tuple[List[List[int]], np.array, np.array, np.array]:
        """
        为文档中的每个词重新采样 topic，并只更新 n_doc_topic
        NOTE: 在 inference 过程中，phi 是固定的，也就是说 n_topic_term，n_topic_term_sum 不发生改变

        Arguments:
            docs {List[List[int]]} -- [预处理后的 docs]
            Z {List[List[int]]} -- [文档中每个词的主题]
            n_doc_topic {np.array} -- [shape:(M,K)]
            phi {np.array} -- [训练过程中得出的 topic_term 多项式分布的参数]
            alpha {float} -- [doc 狄利克雷分布的超参数]

        Returns:
            Tuple[List[List[int]], np.array, np.array, np.array] -- [采样后文档中每个词的主题，更新后的n_doc_topic，更新后的n_topic_term, 更新后的n_topic_term_sum]
        """
        for m, doc in enumerate(docs):
            for n, t in enumerate(doc):
                k = Z[m][n]  # 当前 word 的主题 k
                # 将当前文档当前单词原topic相关计数减去1
                n_doc_topic[m, k] -= 1
                # 重新计算当前文档当前单词属于每个topic的概率
                doc_topic_probability = n_doc_topic[m, ]+alpha  # K vector
                topic_term_probability = phi[:, t]

                p_z = np.multiply(doc_topic_probability,
                                  topic_term_probability)
                # 按照计算出的分布进行采样
                k = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                Z[m][n] = k
                # 将当前文档当前单词新采样的topic相关计数加上1
                n_doc_topic[m, k] += 1

        return Z, n_doc_topic

    @staticmethod
    def _calculate_error_for_inference(n_doc_topic_pre: np.array, n_doc_topic: np.array) -> float:
        """
        计算上一轮迭代和这一轮迭代 n_doc_topic 的变化度，以衡量是否收敛

        Arguments:
            n_doc_topic_pre {np.array} -- [shape:(M,K)]
            n_doc_topic {np.array} -- [shape:(M,K)]

        Returns:
            float -- [变化度]
        """
        error_doc_topic = np.sum(
            np.abs(n_doc_topic-n_doc_topic_pre))/n_doc_topic_pre.sum()
        return error_doc_topic

    # ---------------------------------------------------------------------------- #
    #                               universal funcion                              #
    # ---------------------------------------------------------------------------- #

    @staticmethod
    def _visualize_error(error_list: List[float], figure_size: Tuple[float, float] = (11.7, 8.27)):
        """
        可视化 error 的变化，以衡量是否收敛

        Arguments:
            error_list {List[float]} -- [每一轮迭代的 error]

        Keyword Arguments:
            figure_size {Tuple[float,float]} -- [生成的图片尺寸] (default: {(11.7, 8.27)})
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': figure_size})
        sns.set(style="darkgrid")
        history = pd.DataFrame({"error": error_list}).reset_index()
        sns.lineplot(x="index", y="error", data=history)
        plt.show()

    @staticmethod
    def _calculate_theta(n_doc_topic: np.array, alpha: float) -> np.array:
        """
        计算参数 theta, 也就是把 n_doc_topic+alpha 行归一化

        Arguments:
            n_doc_topic {np.array} -- [shape:(M,K)]
            alpha {float} -- [ doc 狄利克雷分布的超参数]]

        Returns:
            np.array -- [shape:(M,K)]
        """
        return (n_doc_topic+alpha)/(n_doc_topic+alpha).sum(1).reshape(-1, 1)

    @staticmethod
    def _calculate_phi(n_topic_term: np.array, beta: float) -> np.array:
        """
        计算参数 phi, 也就是把 n_topic_term+beta 行归一化

        Arguments:
            n_topic_term {np.array} -- [shape(K,V)]
            beta {float} -- [ word 狄利克雷分布的超参数]

        Returns:
            np.array -- [shape(K,V)]
        """
        return (n_topic_term+beta)/(n_topic_term+beta).sum(1).reshape(-1, 1)

    @staticmethod
    def _calculate_perplexity(docs: List[List[int]], theta: np.array, phi: np.array) -> float:
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

        probability_doc_topic = theta
        probability_topic_term = phi

        for m, doc in enumerate(docs):
            for n, word in enumerate(doc):
                sum_log_p = sum_log_p + \
                    np.log(
                        np.dot(probability_topic_term[:, word], probability_doc_topic[m, :]))
                sum_n_d = sum_n_d + 1
        return np.exp(sum_log_p/(-sum_n_d))
