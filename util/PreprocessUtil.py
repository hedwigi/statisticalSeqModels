# -*- coding: utf-8 -*-
import numpy as np
import re


class PreprocessUtil(object):

    @staticmethod
    def word_trans(token_list, dic_tok2index):
        """

        :param token_list:
        :param dic_tok2index: (0 for UNK)
        :return: index_ll
        """
        index_ll = [[dic_tok2index[tok]] if tok in dic_tok2index else [0] for tok in token_list]
        return np.array(index_ll)

    @staticmethod
    def get_prob_from_2ddict(dict, a, b, default_val):
        """

        :param dict:
        :param a:
        :param b:
        :param default_val:
        :return:
        """
        if a in dict and b in dict[a]:
            return dict[a][b]
        return default_val

    @staticmethod
    def get_max_path_val(dict_paths):
        """

        :param dict_paths:
        :return:
        """
        max_path = None
        max_val = -1*float("infinity")
        for path, val in dict_paths.items():
            if val > max_val:
                max_path = path
                max_val = val
        return max_path, max_val

    @staticmethod
    def process_data(filename):
        """

        :param filename:
        :return: ll_tokindex: list of np.array
                             [[[0], [2], [108]],
                              [], ...]
                ll_stateindex: list of np.array
                            [[0 1 2],
                             [], ...]
                tok2index: {'与':29, "是": 4, ...}
                state2index: {"B": 0, "S": 1, ...}
        """
        ll_tokindex = []
        ll_stateindex = []

        tok2index = {}
        state2index = {}

        index2tok = {}
        index2state = {}

        tok2index["UNK"] = 0

        iter = PreprocessUtil.file_iter(filename)
        sent = iter.__next__()
        while sent:
            l_tokindex = []
            l_stateindex = []
            for tokline in sent:
                tok, _, state = tokline.strip().split("\t")

                # update tok2index
                if tok not in tok2index:
                    tok2index[tok] = len(tok2index)
                    index2tok[tok2index[tok]] = tok

                # update state2index
                if state not in state2index:
                    state2index[state] = len(state2index)
                    index2state[state2index[state]] = state

                l_tokindex.append([tok2index[tok]])
                l_stateindex.append(state2index[state])

            ll_tokindex.append(np.array(l_tokindex))
            ll_stateindex.append(np.array(l_stateindex))

            sent = iter.__next__()

        return ll_tokindex, ll_stateindex, tok2index, state2index, index2tok, index2state

    @staticmethod
    def parse_word_tag(wt):
        wt = re.sub(r'(^\[|\][a-zA-Z0-9]+$)', "", wt)
        return wt.rsplit("/")

    @staticmethod
    def file_iter(file_test):
        """

        :param file_test:
        :return:
        """
        sent = []
        with open(file_test, "r") as fin:
            for line in fin:
                if re.match(r'\s*$', line):
                    yield sent
                    sent = []
                else:
                    sent.append(line)
        yield None

    @staticmethod
    def prob_dict2array(choice_list, prob_dict, default_prob):
        """

        :param choice_list:
        :param prob_dict:
        :param default_prob:
        :return: np.array((len(choice_list)))
        """
        # n_default = len(choice_list) - len(prob_dict)
        # sum_total = float(n_default * default_prob + sum(prob_dict.values()))

        # return [prob_dict[ch]/sum_total if ch in prob_dict else default_prob/sum_total for ch in choice_list]
        return [prob_dict[ch] if ch in prob_dict else 0 for ch in choice_list]