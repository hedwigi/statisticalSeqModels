from math import log
import numpy as np
from util.PreprocessUtil import PreprocessUtil
from algos.BaseMM import BaseMM


class HMM(BaseMM):
    prior_prob = None           # dict, [n_state,]
    transition_prob = None      # dict of dict, [n_state, n_state]
    emission_prob = None        # dict of dict, [n_state, n_obs]

    default_emission_prob = None
    default_prior_prob = None
    default_transition_prob = None

    states = None
    obs_vocab = None

    def train(self, file_train):
        """

        :param file_train: path file
        """
        self.prior_prob = {}
        self.transition_prob = {}
        self.emission_prob = {}

        self.obs_vocab = set([])
        self.states = set([])

        iter = PreprocessUtil.file_iter(file_train)
        sent = iter.__next__()
        while sent:
            prev_state = None
            for i, tokconll in enumerate(sent):
                obs, _, state = tokconll.strip().split("\t")

                self.obs_vocab.add(obs)
                self.states.add(state)

                # update prior_prob2
                # if i == 0:
                #     if state not in self.prior_prob:
                #         self.prior_prob[state] = 0
                #     self.prior_prob[state] += 1

                # update prior_prob
                if state not in self.prior_prob:
                    self.prior_prob[state] = 0
                self.prior_prob[state] += 1

                # update transition_prob
                if prev_state:
                    if prev_state not in self.transition_prob:
                        self.transition_prob[prev_state] = {}
                    if state not in self.transition_prob[prev_state]:
                        self.transition_prob[prev_state][state] = 0
                    self.transition_prob[prev_state][state] += 1

                # update emission_prob
                if state not in self.emission_prob:
                    self.emission_prob[state] = {}
                if obs not in self.emission_prob[state]:
                    self.emission_prob[state][obs] = 0
                self.emission_prob[state][obs] += 1

                prev_state = state

            sent = iter.__next__()

        # finalize prior_prob
        sum_ = sum(self.prior_prob.values())
        for s in self.prior_prob:
            self.prior_prob[s] /= sum_

        self.default_prior_prob = 1/len(self.prior_prob)

        # finalize transition_prob
        for s1, d in self.transition_prob.items():
            sum_s1 = sum(d.values())
            for s2 in d:
                d[s2] /= sum_s1

        self.default_transition_prob = 1/len(self.prior_prob)

        # finalize emission_prob
        for s, d in self.emission_prob.items():
            sum_s = sum(d.values())
            for o in d:
                d[o] /= sum_s

        self.default_emission_prob = 1/len(self.obs_vocab)

    def viterbi(self, token_list):
        """

        :param token_list:
        :return:
        """
        # step 0
        prev_step = {}
        for state, prior in self.prior_prob.items():
            emission_prob = self.default_emission_prob
            if state in self.emission_prob and token_list[0] in self.emission_prob[state]:
                emission_prob = self.emission_prob[state][token_list[0]]
            prev_step[state] = log(prior) + log(emission_prob)

        # iteration
        for tok in token_list[1:]:
            current_step = {}
            for current_state in self.states:
                paths2current_state = {}
                for prev_path in prev_step:
                    prev_state = prev_path[-1] if isinstance(prev_path, tuple) else prev_path
                    newpath= tuple(list(prev_path) + [current_state]) if isinstance(prev_path, tuple) else (prev_path, current_state)

                    paths2current_state[newpath] = prev_step[prev_path] \
                                                   + log(PreprocessUtil.get_prob_from_2ddict(self.transition_prob, prev_state, current_state, self.default_transition_prob)) \
                                                   + log(PreprocessUtil.get_prob_from_2ddict(self.emission_prob, current_state, tok, self.default_emission_prob))
                maxpath, maxval = PreprocessUtil.get_max_path_val(paths2current_state)
                current_step[maxpath] = maxval
            prev_step = current_step

        max_final_path, max_final_val = PreprocessUtil.get_max_path_val(prev_step)
        return list(max_final_path)

    def generate_seq(self, seq_length, initial_state=None):
        """

        :param seq_length:
        :param initial_state:
        :return:
        """
        assert self.emission_prob is not None
        assert self.transition_prob is not None
        assert self.prior_prob is not None

        observations = []
        states = []
        transitions = []
        emissions = []

        if initial_state:
            prev_state = initial_state
        else:
            prior_prob_array = PreprocessUtil.prob_dict2array(self.states, self.prior_prob, self.default_prior_prob)
            prev_state = np.random.choice(list(self.states), 1, p=prior_prob_array)[0]  # 采样初始状态

        obs = np.random.choice(list(self.emission_prob[prev_state].keys()), 1,
                                                p=PreprocessUtil.prob_dict2array(list(self.emission_prob[prev_state].keys()),
                                                                                 self.emission_prob[prev_state],
                                                                                 self.default_emission_prob))[0]  # 采样得到序列第一个值
        states.append(prev_state)
        observations.append(obs)
        transitions.append(self.prior_prob[prev_state])
        emissions.append(self.emission_prob[prev_state][obs])

        for i in range(1, seq_length):
            # P(Zn+1)=P(Zn+1|Zn)P(Zn)
            current_state = np.random.choice(list(self.transition_prob[prev_state].keys()), 1, p=PreprocessUtil.prob_dict2array(list(self.transition_prob[prev_state].keys()),
                                                                                                    self.transition_prob[prev_state],
                                                                                                    self.default_transition_prob))[0]
            current_obs = np.random.choice(list(self.emission_prob[current_state].keys()), 1, p=PreprocessUtil.prob_dict2array(list(self.emission_prob[current_state].keys()),
                                                                  self.emission_prob[current_state],
                                                                  self.default_emission_prob))[0]
            # P(Xn+1|Zn+1)
            observations.append(current_obs)
            states.append(current_state)
            transitions.append(self.transition_prob[prev_state][current_state])
            emissions.append(self.emission_prob[current_state][current_obs])

            prev_state = current_state

        return observations, states, transitions, emissions
