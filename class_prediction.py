import fasttext
import pandas as pd
import numpy as np
import pickle
import argparse


def load_mol_frag_model(model_path):
    return fasttext.load_model(model_path)


def load_kmeans_model(kmenas_path):
    with open(kmenas_path, 'rb') as handle:
        return pickle.load(handle)


def get_mol_vec(frag_vec_model, frag):
    """
    get molecular vector by sum all fragment vectors
    :return:
    """
    frags = frag.split(',')
    frag2vec = []
    for f in frags:
        f_v = frag_vec_model.get_word_vector(f)
        frag2vec.append(f_v)
    frag2vec = np.array(frag2vec)
    # print(frag2vec)
    mol_vec = np.sum(frag2vec, axis=0)
    return mol_vec


def predict_class():
    """
    predict class,
    :return:
    """
    pass


def find_nn():
    """
    find top 10 nearest neighbors in all training set (more than 10,000,000 molecules)
    :return:
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='molecule class prediction and finding nearest neighbors')
    parser.add_argument('frag_vec_model_fp',
                        help='file path of trained fragment vector model')
    parser.add_argument('kmeans_model_fp', help='file path of trained kmeans model')
    parser.add_argument('input_fp', help='file path of step2 result')
    parser.add_argument('--result_fp',
                        help='result file path')
    parser.add_argument('--topn', default=10, help='find top N of nearest neighbors')
    parser.add_argument('--log_fn',
                        help='log file name')

    args = parser.parse_args()
    input_fp = args.input_fp
    frag_vec_model_fp = args.frag_vec_model_fp
    kmeans_model_fp = args.kmeans_model_fp
    result_fp = args.result_fp
    topn = args.topn
    log_fp = args.log_fn

    frag_raw_info = pd.read_csv(input_fp, index_col=0, sep='\t', header=None)
    frag_raw_info = frag_raw_info.loc[:, [3]].copy()
    print(frag_raw_info.head())
    cid2mol_vec = {}
    frag_vec_model = load_mol_frag_model(frag_vec_model_fp)

    for i in frag_raw_info.index:
        _frag = str(frag_raw_info.loc[i, 3])
        mol_vec = get_mol_vec(frag_vec_model, _frag)
        cid2mol_vec[i] = mol_vec

    mol_vec_df = pd.DataFrame.from_dict(data=cid2mol_vec, orient='index')
    mol_vec_df.to_csv(result_fp)

