from util.PreprocessUtil import PreprocessUtil
import os


def transform2conll(infile, basename):
    """
        BI0
    :param infile:
                运动/v 开始/v 后/d ,/w
                刘翔/nr 说/v ，/w
    :param basename:
                [SEG]
                运 - B
                动 - I
                开 - B
                始 - I
                后 - B
                ， - B

                [POS]
                运动 - B-v
                开始 - B-v
                后 - B-v
                ， - B-w

                [NER]
                刘 - B-nr
                翔 - I-nr
                说 - 0
                ， - 0

    :return:
    """
    with open(basename + ".seg", "w") as fout_seg:
        with open(basename + ".pos", "w") as fout_pos:
            with open(basename + ".ner", "w") as fout_ner:

                with open(infile, "r") as fin:
                    for line in fin:
                        tok_pos_list = [PreprocessUtil.parse_word_tag(wp) for wp in line.strip().split()]

                        write_conll(tok_pos_list, "seg", fout_seg)
                        write_conll(tok_pos_list, "pos", fout_pos)
                        write_conll(tok_pos_list, "ner", fout_ner)


def write_conll(tok_pos_list, type, fout):
    """

    :param tok_pos_list: for one line
    :param fout:
    :return:
    """
    lineconll = []
    for tok, pos in tok_pos_list:
        if type == "seg":
            lineconll += tok2segconll(tok)
        elif type == "pos":
            lineconll += tok2posconll(tok, pos)
        elif type == "ner":
            lineconll += tok2nerconll(tok, pos)
        else:
            raise ValueError("no type %s" % type)

    fout.write("\n".join(lineconll))
    fout.write("\n\n")


def tok2segconll(tok):
    """
        BIS

    :param tok: 与此同时
    :return: list of string
            与 - B
            此 - I
            同 - I
            时 - I
    """
    if len(tok) == 0:
        return []

    feat_col = "\t-\t"
    if len(tok) == 1:
        return [tok + feat_col + "B"]

    tokconll = []
    for i, char in enumerate(list(tok)):
        if i == 0:
            tokconll += [char + feat_col + "B"]
        else:
            tokconll += [char + feat_col + "I"]
    return tokconll


def tok2posconll(tok, pos):
    """

    :param tok: 运动
    :param pos:
    :return:
            运动 - v
    """
    if len(tok) == 0:
        return []

    feat_col = "\t-\t"
    return [tok + feat_col + "B-" + pos]


def tok2nerconll(tok, pos):
    """
        BISO * (nr, t, ns)

    :param tok:
    :param pos:
    :return:
    """
    if len(tok) == 0:
        return []

    ner_tags = ["nr", "t", "ns"]
    feat_col = "\t-\t"
    if len(tok) == 1:
        tag = "B-" + pos if pos in ner_tags else "0"
        return [tok + feat_col + tag]

    tokconll = []
    for i, char in enumerate(list(tok)):
        if i == 0:
            tag = "B-" + pos if pos in ner_tags else "0"
            tokconll += [char + feat_col + tag]
        else:
            tag = "I-" + pos if pos in ner_tags else "0"
            tokconll += [char + feat_col + tag]
    return tokconll


if __name__ == "__main__":

    base_filename = "train"
    infile = os.path.join(os.path.dirname(__file__), "orig/" + base_filename)
    outfile = os.path.join(os.path.dirname(__file__), base_filename)

    transform2conll(infile, outfile)
