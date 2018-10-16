import time
from abc import ABCMeta, abstractmethod
from util.PreprocessUtil import PreprocessUtil


class BaseMM(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, file_train):
        pass

    def predict_file(self, pathin, pathout):
        """

        :param pathin: conll format
        :param pathout: conll format, add predict results to the last column
        :return:
        """

        iter = PreprocessUtil.file_iter(pathin)
        sent_conll = iter.__next__()
        n = 0
        with open(pathout, "w") as fout:
            while sent_conll:
                n += 1
                if n % 50 == 0:
                    print(n)
                tok_list = []
                for tok_conll in sent_conll:
                    tok, _, true_state = tok_conll.strip().split("\t")
                    tok_list.append(tok)
                state_list_pred = self.viterbi(tok_list)
                for i, tok_conll in enumerate(sent_conll):
                    fout.write(tok_conll.strip() + "\t" + state_list_pred[i] + "\n")
                fout.write("\n")
                sent_conll = iter.__next__()

    @abstractmethod
    def viterbi(self, token_list):
        pass