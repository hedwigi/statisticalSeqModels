# -*- coding: utf-8 -*-
import numpy as np
from abc import ABCMeta, abstractmethod
from util.PreprocessUtil import PreprocessUtil


class _BaseHMMem(object):
    """
    基本HMM虚类，需要重写关于发射概率的相关虚函数
    n_state : 隐藏状态的数目
    n_iter : 迭代次数
    x_size : 观测值维度
    start_prob : 初始概率
    transmat_prob : 状态转换概率
    """
    __metaclass__ = ABCMeta  # 虚类声明

    def __init__(self, n_state=1, x_size=1, iter=20):
        self.n_state = n_state
        self.observation_size = x_size
        self.start_prob = np.ones(n_state) * (1.0 / n_state)  # 初始状态概率
        self.transmat_prob = np.ones((n_state, n_state)) * (1.0 / n_state)  # 状态转换概率矩阵
        self.trained = False  # 是否需要重新训练
        self.n_iter = iter  # EM训练的迭代次数

    # 初始化发射参数
    @abstractmethod
    def _init(self, X):
        pass

    # 虚函数：返回发射概率
    @abstractmethod
    def emit_prob(self, x):  # 求x在状态k下的发射概率 P(X|Z)
        return np.array([0])

    # 虚函数
    @abstractmethod
    def generate_x(self, z):  # 根据隐状态生成观测值x p(x|z)
        return np.array([0])

    # 虚函数：发射概率的更新
    @abstractmethod
    def emit_prob_updated(self, X, post_state):
        pass

    # 通过HMM生成序列
    def generate_seq(self, seq_length, index2tok, index2state):
        """

        :param seq_length:
        :param index2tok:
        :param index2state:
        :return:
        """
        lidx_observations = np.zeros(seq_length)
        lidx_states = np.zeros(seq_length)

        prev_state = np.random.choice(self.n_state, 1, p=self.start_prob)  # 采样初始状态
        lidx_observations[0] = self.generate_x(prev_state)  # 采样得到序列第一个值
        lidx_states[0] = prev_state

        for i in range(1, seq_length):
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            current_state = np.random.choice(self.n_state, 1, p=self.transmat_prob[prev_state, :][0])

            # P(Xn+1|Zn+1)
            lidx_observations[i] = self.generate_x(current_state)
            lidx_states[i] = current_state

            prev_state = current_state

        observations = [index2tok[io] for io in lidx_observations]
        states = [index2state[ist] for ist in lidx_states]

        return lidx_observations, lidx_states, observations, states

    # 估计序列X出现的概率
    def X_prob(self, X, Z_seq=np.array([])):
        # 状态序列预处理
        # 判断是否已知隐藏状态
        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向后传递因子
        _, c = self.forward(X, Z)  # P(x,z)
        # 序列的出现概率估计
        prob_X = np.sum(np.log(c))  # P(X)
        return prob_X

    # 已知当前序列预测未来（下一个）观测值的概率
    def predict(self, X, x_next, Z_seq=np.array([]), istrain=True):
        if self.trained == False or istrain == False:  # 需要根据该序列重新训练
            self.train(X)

        X_length = len(X)
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))
        # 向前向后传递因子
        alpha, _ = self.forward(X, Z)  # P(x,z)
        prob_x_next = self.emit_prob(np.array([x_next])) * np.dot(alpha[X_length - 1], self.transmat_prob)
        return prob_x_next

    def decode(self, X, istrained=True):
        """
        利用维特比算法，已知序列求其隐藏状态值
        :param X: 观测值序列
        :param istrained: 是否根据该序列进行训练
        :return: 隐藏状态序列
        """
        if self.trained is False or istrained is False:  # 需要根据该序列重新训练
            self.train(X)

        X_length = len(X)  # 序列长度
        state = np.zeros(X_length)  # 隐藏状态

        pre_state = np.zeros((X_length, self.n_state))  # 保存转换到当前隐藏状态的最可能的前一状态
        max_pro_state = np.zeros((X_length, self.n_state))  # 保存传递到序列某位置当前状态的最大概率

        _, c = self.forward(X, np.ones((X_length, self.n_state)))
        max_pro_state[0] = self.emit_prob(X[0]) * self.start_prob * (1 / c[0])  # 初始概率

        # 前向过程
        for i in range(X_length):
            if i == 0: continue
            for k in range(self.n_state):
                prob_state = self.emit_prob(X[i])[k] * self.transmat_prob[:, k] * max_pro_state[i - 1]
                max_pro_state[i][k] = np.max(prob_state) * (1 / c[i])
                pre_state[i][k] = np.argmax(prob_state)

        # 后向过程
        state[X_length - 1] = np.argmax(max_pro_state[X_length - 1, :])
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            state[i] = pre_state[i + 1][int(state[i + 1])]

        return state

    # 针对于多个序列的训练问题
    def train_batch(self, ll_charindex, ll_stateindex=list()):
        """

        :param ll_charindex: list of list of list，
                             [[[0], [2], [108]],
                              [], ...]
        :param ll_stateindex: list of list of int，默认为空列表（即未知隐状态情况）
                            [[0 1 2],
                             [], ...]
        :return:
        """
        # 针对于多个序列的训练问题，其实最简单的方法是将多个序列合并成一个序列，而唯一需要调整的是初始状态概率
        self.trained = True
        batch_size = len(ll_charindex)  # 序列个数
        self._init(self.expand_list(ll_charindex))  # 发射概率的初始化

        # 状态序列预处理，将单个状态转换为1-to-k的形式
        # 判断是否已知隐藏状态
        if ll_stateindex == list():
            Z = []  # 初始化状态序列list
            for n in range(batch_size):
                Z.append(list(np.ones((len(ll_charindex[n]), self.n_state))))
        else:
            Z = []
            for n in range(batch_size):
                Z.append(np.zeros((len(ll_charindex[n]), self.n_state)))
                for i in range(len(Z[n])):
                    Z[n][i][int(ll_stateindex[n][i])] = 1

        for e in range(self.n_iter):  # EM步骤迭代
            # 更新初始概率过程
            #  E步骤
            print("iter: ", e)
            b_post_state = []  # 批量累积：状态的后验概率，类型list(array)
            b_post_adj_state = np.zeros((self.n_state, self.n_state))  # 批量累积：相邻状态的联合后验概率，数组
            b_start_prob = np.zeros(self.n_state)  # 批量累积初始概率
            for n in range(batch_size):  # 对于每个序列的处理
                X_length = len(ll_charindex[n])
                alpha, c = self.forward(ll_charindex[n], Z[n])  # P(x,z)
                beta = self.backward(ll_charindex[n], Z[n], c)  # P(x|z)

                post_state = alpha * beta / np.sum(alpha * beta)  # 归一化！
                b_post_state.append(post_state)
                post_adj_state = np.zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
                for i in range(X_length):
                    if i == 0: continue
                    if c[i] == 0: continue
                    post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                            beta[i] * self.emit_prob(ll_charindex[n][i])) * self.transmat_prob

                if np.sum(post_adj_state) != 0:
                    post_adj_state = post_adj_state / np.sum(post_adj_state)  # 归一化！
                b_post_adj_state += post_adj_state  # 批量累积：状态的后验概率
                b_start_prob += b_post_state[n][0]  # 批量累积初始概率

            # M步骤，估计参数，最好不要让初始概率都为0出现，这会导致alpha也为0
            b_start_prob += 0.001 * np.ones(self.n_state)
            self.start_prob = b_start_prob / np.sum(b_start_prob)
            b_post_adj_state += 0.001
            for k in range(self.n_state):
                if np.sum(b_post_adj_state[k]) == 0: continue
                self.transmat_prob[k] = b_post_adj_state[k] / np.sum(b_post_adj_state[k])

            self.emit_prob_updated(self.expand_list(ll_charindex), self.expand_list(b_post_state))

    def expand_list(self, X):
        # 将list(array)类型的数据展开成array类型
        C = []
        for i in range(len(X)):
            C += list(X[i])
        return np.array(C)

    # 针对于单个长序列的训练
    def train(self, X, Z_seq=np.array([])):
        # 输入X类型：array，数组的形式
        # 输入Z类型: array，一维数组的形式，默认为空列表（即未知隐状态情况）
        self.trained = True
        X_length = len(X)
        self._init(X)

        # 状态序列预处理
        # 判断是否已知隐藏状态
        if Z_seq.any():
            Z = np.zeros((X_length, self.n_state))
            for i in range(X_length):
                Z[i][int(Z_seq[i])] = 1
        else:
            Z = np.ones((X_length, self.n_state))

        for e in range(self.n_iter):  # EM步骤迭代
            # 中间参数
            print(e, " iter")
            # E步骤
            # 向前向后传递因子
            alpha, c = self.forward(X, Z)  # P(x,z)
            beta = self.backward(X, Z, c)  # P(x|z)

            post_state = alpha * beta
            post_adj_state = np.zeros((self.n_state, self.n_state))  # 相邻状态的联合后验概率
            for i in range(X_length):
                if i == 0: continue
                if c[i] == 0: continue
                post_adj_state += (1 / c[i]) * np.outer(alpha[i - 1],
                                                        beta[i] * self.emit_prob(X[i])) * self.transmat_prob

            # M步骤，估计参数
            self.start_prob = post_state[0] / np.sum(post_state[0])
            for k in range(self.n_state):
                self.transmat_prob[k] = post_adj_state[k] / np.sum(post_adj_state[k])

            self.emit_prob_updated(X, post_state)

    # 求向前传递因子
    def forward(self, X, Z):
        X_length = len(X)
        alpha = np.zeros((X_length, self.n_state))  # P(x,z)
        alpha[0] = self.emit_prob(X[0]) * self.start_prob * Z[0]  # 初始值
        # 归一化因子
        c = np.zeros(X_length)
        c[0] = np.sum(alpha[0])
        alpha[0] = alpha[0] / c[0]
        # 递归传递
        for i in range(X_length):
            if i == 0: continue
            alpha[i] = self.emit_prob(X[i]) * np.dot(alpha[i - 1], self.transmat_prob) * Z[i]
            c[i] = np.sum(alpha[i])
            if c[i] == 0: continue
            alpha[i] = alpha[i] / c[i]

        return alpha, c

    # 求向后传递因子
    def backward(self, X, Z, c):
        X_length = len(X)
        beta = np.zeros((X_length, self.n_state))  # P(x|z)
        beta[X_length - 1] = np.ones((self.n_state))
        # 递归传递
        for i in reversed(range(X_length)):
            if i == X_length - 1: continue
            beta[i] = np.dot(beta[i + 1] * self.emit_prob(X[i + 1]), self.transmat_prob.T) * Z[i]
            if c[i + 1] == 0: continue
            beta[i] = beta[i] / c[i + 1]

        return beta


class HMMem(_BaseHMMem):
    """
    发射概率为离散分布的HMM
    参数：
    emit_prob : 离散概率分布
    observation_size：表示观测值的种类
    此时观测值大小observation_size默认为1
    """

    def __init__(self, n_state=1, observation_size=1, iter=20):
        _BaseHMMem.__init__(self, n_state=n_state, x_size=1, iter=iter)
        self.emission_prob = np.ones((n_state, observation_size)) * (1.0 / observation_size)  # 初始化发射概率均值
        self.observation_size = observation_size

    def _init(self, X):
        self.emission_prob = np.random.random(size=(self.n_state, self.observation_size))
        for k in range(self.n_state):
            self.emission_prob[k] = self.emission_prob[k] / np.sum(self.emission_prob[k])

    def emit_prob(self, x):  # 求x在状态k下的发射概率
        prob = np.zeros(self.n_state)
        for i in range(self.n_state): prob[i] = self.emission_prob[i][int(x[0])]
        return prob

    def generate_x(self, z):  # 根据状态生成x p(x|z)
        self.normalize_emission_prob()
        return np.random.choice(self.observation_size, 1, p=self.emission_prob_norm[z][0])

    def normalize_emission_prob(self):
        row_sum = self.emission_prob.sum(axis=1)[:, None]
        self.emission_prob_norm = self.emission_prob/row_sum

    def emit_prob_updated(self, X, post_state):  # 更新发射概率
        self.emission_prob = np.zeros((self.n_state, self.observation_size))
        X_length = len(X)
        for n in range(X_length):
            self.emission_prob[:, int(X[n])] += post_state[n]

        self.emission_prob += 0.1 / self.observation_size
        for k in range(self.n_state):
            if np.sum(post_state[:, k]) == 0: continue
            self.emission_prob[k] = self.emission_prob[k] / np.sum(post_state[:, k])

    def predict_file(self, pathin_test, pathout_hmm_pred, tok2index, index2state):
        """

        :param pathin_test:
        :param pathout_hmm_pred:
        :param tok2index:
        :param index2state:
        :return:
        """
        iter = PreprocessUtil.file_iter(pathin_test)
        sent_conll = iter.__next__()
        with open(pathout_hmm_pred, "w") as fout:
            while sent_conll:

                tok_list = []
                for tok_conll in sent_conll:
                    tok, _, true_state = tok_conll.strip().split("\t")
                    tok_list.append(tok)

                state_list_pred = self.decode(PreprocessUtil.word_trans(tok_list, tok2index))
                for i, tok_conll in enumerate(sent_conll):
                    fout.write(tok_conll.strip() + "\t" + index2state[int(state_list_pred[i])] + "\n")
                fout.write("\n")
                sent_conll = iter.__next__()
