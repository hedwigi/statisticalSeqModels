from nltk.classify import MaxentClassifier
import time
from math import log
from util.PreprocessUtil import PreprocessUtil
from algos.BaseMM import BaseMM


class MEMM(BaseMM):

    classifier = None
    states = None
    zero_state = "<S>"
    max_iter = 1

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def train(self, file_train):
        """

        :param file_train:
        :return:
        """
        self.states = set([])

        # me classifier
        labeled_featuresets = []  # list of (feature_dict, lable)

        iter = PreprocessUtil.file_iter(file_train)
        sent = iter.__next__()
        while sent:
            prev_state = self.zero_state
            for tokconll in sent:
                obs, _, state = tokconll.strip().split("\t")

                self.states.add(state)

                feature_dict = {"prev_state": prev_state, "obs": obs}
                labeled_featuresets.append((feature_dict, state))
                prev_state = state

            sent = iter.__next__()

        self.classifier = MaxentClassifier.train(labeled_featuresets, max_iter=self.max_iter)

    def viterbi(self, token_list):
        """

        :param token_list:
        :return:
        """
        # step 0
        prev_step = {}
        for state in self.states:
            feature_dict = {"prev_state": self.zero_state, "obs": token_list[0]}
            prev_step[state] = log(self.classifier.prob_classify(feature_dict).prob(state))

        # iteration
        for tok in token_list[1:]:
            current_step = {}
            for current_state in self.states:
                paths2current_state = {}
                for prev_path in prev_step:
                    prev_state = prev_path[-1] if isinstance(prev_path, tuple) else prev_path
                    newpath = tuple(list(prev_path) + [current_state]) if isinstance(prev_path, tuple) else (
                    prev_path, current_state)
                    feature_dict = {"prev_state": prev_state, "obs": tok}
                    paths2current_state[newpath] = prev_step[prev_path] \
                                                   + log(self.classifier.prob_classify(feature_dict).prob(current_state))
                maxpath, maxval = PreprocessUtil.get_max_path_val(paths2current_state)
                current_step[maxpath] = maxval
            prev_step = current_step
        max_final_path, max_final_val = PreprocessUtil.get_max_path_val(prev_step)
        return list(max_final_path)
