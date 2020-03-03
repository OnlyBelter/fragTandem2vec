"""
step3: Training molFrag2vec model by FastText, and get the vectors of all fragments.
# usage:
$ python training_model.py ./demo_data/step2_frag_id_sentence.csv ./demo_data/step3_molFrag2vec_demo.bin

"""

import fasttext
import argparse
import pandas as pd
import os


def train_model(train_x_fp, model_fp):
    """

    :param train_x_fp: file path of training set, only fragments separated by space for one line one molecule
    :param model_fp: file path of saving model
    :return:
    """
    model = fasttext.train_unsupervised(train_x_fp, dim=150, ws=10, thread=4, epoch=5)
    model.save_model(model_fp)


def get_frag_vector(model_fp, frag_id2vec_fp):
    """
    get fragment vector from model
    :param model_fp: file path of trained model
    :param frag_id2vec_fp: file path of frag_id2vec
    :return:
    """
    model = fasttext.load_model(model_fp)
    words = model.words
    frag2vec = {}
    for f in words:
        frag2vec[f] = model.get_word_vector(f)
    frag2vec_df = pd.DataFrame.from_dict(frag2vec, orient='index')
    print('>>> There are {} fragments in total.'.format(frag2vec_df.shape[0]))
    frag2vec_df.to_csv(frag_id2vec_fp, index_label='frag_id')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training molFrag2vec model using FastText')
    parser.add_argument('input_fn',
                        help='training set file path')
    parser.add_argument('result_fn',
                        help='where to save trained model')
    args = parser.parse_args()
    input_file = args.input_fn
    result_fn = args.result_fn
    train_model(input_file, result_fn)
    result_dir = os.path.dirname(result_fn)
    frag2vec_fp = os.path.join(result_dir, 'step3_frag2vec_model_fragTandem2vec.csv')
    # mol2vec_fp = os.path.join(result_dir, 'selected_mol2vec.csv')
    get_frag_vector(result_fn, frag_id2vec_fp=frag2vec_fp)
