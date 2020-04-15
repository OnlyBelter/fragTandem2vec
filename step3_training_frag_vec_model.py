"""
step3: Training molFrag2vec model by FastText, and get the vectors of all fragments.
# usage: without supervised training
$ python step3_training_frag_vec_model.py big-data/moses_dataset/result/step2_parallel_frag_smiles_sentence.csv big-data/moses_dataset/result/
 --model_fn step3_model_parallel2vec.bin --frag_vec_fn step3_model_parallel2vec.csv

# usage: with supervised training
$ python step3_training_frag_vec_model.py big-data/moses_dataset/result/step2_parallel_frag_smiles_sentence.csv big-data/moses_dataset/result/
 --model_fn step3_model_parallel2vec.bin --frag_vec_fn step3_model_parallel2vec.csv --supervised_training
"""

import fasttext
import argparse
import pandas as pd
import os
from pub_func import get_format_time
from helper_func import cal_md_by_smiles, get_class, vis_class, SuperviseClassModel, get_xy


def train_model(train_x_fp, model_fp, ws=5, minn=0, maxn=0, epoch=10):
    """

    :param train_x_fp: file path of training set, only fragments separated by space for one line one molecule
    :param model_fp: file path of saving model
    :return:
    """
    # parameters in my short thesis
    # model = fasttext.train_unsupervised(train_x_fp, dim=150, ws=10, thread=4, epoch=5)
    model = fasttext.train_unsupervised(train_x_fp, model='cbow', dim=100, ws=ws,
                                        thread=4, epoch=epoch, minCount=3, minn=minn, maxn=maxn)
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
    frag2vec_df.to_csv(frag_id2vec_fp, index_label='fragment')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training molFrag2vec model using FastText')
    parser.add_argument('input_fn', help='training set file path')
    parser.add_argument('result_dir', help='the directory of result files')
    parser.add_argument('--model_fn', help='where to save trained model')
    parser.add_argument('--frag_vec_fn', help='where to save fragment vector')
    # parser.add_argument('--frag2num_fp', help='the file path of frag2num from step2', default='no_input')
    parser.add_argument('--supervised_training', action='store_true', default=False,
                        help='if train fragment vectors by supervised method to reorganize vector space.')
    args = parser.parse_args()
    input_file = args.input_fn
    result_dir = args.result_dir
    model_fn = args.model_fn
    frag_vec_fn = args.frag_vec_fn
    supervised_training = args.supervised_training
    # frag2num_fp = args.frag2num_fp
    model_name = model_fn.replace('.csv', '').split('_')[-1]
    model_fp = os.path.join(result_dir, model_fn)
    frag2vec_fp = os.path.join(result_dir, frag_vec_fn)

    # t0 = get_format_time()
    # print('  >Start to train vector model in {}...'.format(t0))
    # train_model(input_file, model_fp)
    # # mol2vec_fp = os.path.join(result_dir, 'selected_mol2vec.csv')
    # get_frag_vector(model_fp, frag_id2vec_fp=frag2vec_fp)
    # t1 = get_format_time()
    # print('  >Finished training vector model in {}...'.format(t1))

    if supervised_training:
        # if frag2num_fp == 'no_input':
        #     raise Exception('You must give the file path of file frag2num from step2 by parameter --frag2num_fp, '
        #                     'since supervised_training is open.')
        # get fragment information
        print('  >Start to get supervise-trained fragment vectors...')
        frag2vec = pd.read_csv(frag2vec_fp, index_col='fragment')
        frag_smiles_list = frag2vec.index.to_list()
        frag_smiles_list = [i for i in frag_smiles_list if i != '</s>']  # remove non-SMILES holder
        frag_info = cal_md_by_smiles(smiles_list=frag_smiles_list)

        frag_info.to_csv(os.path.join(result_dir, 'step3_model_{}_frag_info.csv'.format(model_name)),
                         index_label='fragment')
        # x = frag2vec.values
        frag2class = get_class(frag_info, min_number=1)
        xy_result = get_xy(frag2vec=frag2vec, frag2class=frag2class)
        x = xy_result['x']
        y = xy_result['y']  # one hot array
        n_class = y.shape[1]
        print('  >number of class is: {}'.format(n_class))
        print('  > type of x: {}'.format(type(x)))
        # training fragment vector
        supervised_model = SuperviseClassModel(n_output=n_class)
        supervised_model.model_compile()
        supervised_model.training(x=x, y=y)
        frag2vec_new = supervised_model.get_embedding_vec(x)
        frag2vec_new_df = pd.DataFrame(data=frag2vec_new, index=x.index)
        frag2vec_new_df.to_csv(os.path.join(result_dir, 'step3_model_{}_supervise_trained.csv'.format(model_name)))
