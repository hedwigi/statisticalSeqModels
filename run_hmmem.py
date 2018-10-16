import os

from algos.HMMem import HMMem
from util.PreprocessUtil import PreprocessUtil

if __name__ == "__main__":
    task = "ner"  # seg|pos|ner
    max_iter = 5

    file_train = os.path.join(os.path.dirname(__file__), "data/train." + task)
    pathin_test = os.path.join(os.path.dirname(__file__), "data/test." + task)

    pathout_hmm_pred = os.path.join(os.path.dirname(__file__), "data/pred_results/hmmem/pred." + task)

    ll_tokindex, ll_stateindex, \
    tok2index, state2index, \
    index2tok, index2state = PreprocessUtil.process_data(file_train)

    hmm = HMMem(len(state2index), len(tok2index), max_iter)
    hmm.train_batch(ll_tokindex, ll_stateindex)
    hmm.predict_file(pathin_test, pathout_hmm_pred, tok2index, index2state)

    # generate sequence
    lidx_observations, lidx_states, observations, states = hmm.generate_seq(5, index2tok, index2state)
    print(lidx_observations)
    print(lidx_states)
    print(observations)
    print(states)
