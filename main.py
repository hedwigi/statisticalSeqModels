import os
import time
from algos.HMM import HMM
from algos.MEMM import MEMM


if __name__ == "__main__":

    # Parameters
    task = "ner"  # seg|pos|ner
    model = "memm" # memm|hmm

    file_train = os.path.join(os.path.dirname(__file__), "data/train." + task)
    file_test = os.path.join(os.path.dirname(__file__), "data/test." + task)
    pathout = os.path.join(os.path.dirname(__file__), "data/pred_results/" + model + "/pred." + task)

    if model == "memm":
        model = MEMM(5)
    elif model == "hmm":
        model = HMM()
    else:
        raise ValueError("no model named %s" % model)

    st = time.clock()
    model.train(file_train)
    print("training: %f s" % (time.clock() - st))

    st = time.clock()
    model.predict_file(file_test, pathout)
    print("predict: %f s" % (time.clock() - st))

    # Generate sequence with hmm
    if model == "hmm":
        initial_state = "B-m"
        # print(model.emission_prob[initial_state])
        # print(len(model.emission_prob[initial_state]))
        for i in range(10):
            observations, states, transitions, emissions = model.generate_seq(5, initial_state)
            line = "\n".join(["\t".join(observations),
                    "\t".join(states),
                    "\t".join([str(round(p, 3)) for p in transitions]),
                    "\t".join([str(round(p, 3)) for p in emissions])])
            print(line + "\n")
