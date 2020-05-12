from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from helper_func import get_class_md_combination, get_mol_vec


def sampling_train_set(mol2md_info_fn, n_max=5000):
    """
    select a subset of training set without some classes (remove 20%)
    :param mol2md_info_fn:
    :param n_max: max samples of sampling in each class
    :return: sampling result, a list of cid
    """
    mol2md_info = pd.read_csv(mol2md_info_fn, index_col='cid')
    mol2md_info = get_class_md_combination(mol2md_info, min_number=1)
    unique_labels = mol2md_info['class'].unique()
    n_80_per = int(np.ceil(len(unique_labels) * 0.9))
    unique_labels_80 = np.random.choice(unique_labels, n_80_per, replace=False)
    small_class_bool = mol2md_info['class'].value_counts() < 10
    small_class = small_class_bool[small_class_bool].index.to_list()
    print('num: {}, small class: {}'.format(len(small_class), small_class))
    unique_labels_80 = set(unique_labels_80) - set(small_class)
    selected_mol2md_info = mol2md_info[mol2md_info['class'].isin(unique_labels_80)].copy()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    for train_inx, test_inx in split.split(selected_mol2md_info, selected_mol2md_info['class']):
        train_set = selected_mol2md_info.iloc[train_inx]
        test_set = selected_mol2md_info.iloc[test_inx]
    other_mol2md_info = mol2md_info[~mol2md_info['class'].isin(unique_labels_80)].copy()
    test_set = other_mol2md_info.append(test_set)
    return {'train_set': train_set, 'test_set': test_set}


def list2dic(a_list):
    a_dic = {}
    for i in a_list:
        a_dic[i] = 0
    return a_dic


def nn_model(x, y, result_dir):
    m_part1 = keras.Sequential([keras.layers.Dense(50, activation='selu', input_shape=[100]),
                                keras.layers.Dense(30, activation='selu')])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(8)])
    model = keras.Sequential([m_part1, m_part2])
    model.compile(optimizer='rmsprop', loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x, y, epochs=10, batch_size=32)
    m_part1.save(os.path.join(result_dir, 'm_part1_2.h5'))
    model.save(os.path.join(result_dir, 'model_2.h5'))


def nn_model_regression(x, y, result_dir):
    m_part1 = keras.Sequential([keras.layers.Dense(50, activation='selu', input_shape=[100]),
                                keras.layers.Dense(30, activation='selu')])
    m_part2 = keras.Sequential([
        keras.layers.Dense(50, activation='selu', input_shape=[30]),
        keras.layers.Dense(100, activation='selu'),
        keras.layers.Dense(8, activation='softplus')])
    model = keras.Sequential([m_part1, m_part2])
    model.compile(optimizer='rmsprop', loss='mse',
                  metrics=['mse'])
    history = model.fit(x, y, epochs=10, batch_size=32)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.to_csv(os.path.join(result_dir, 'history_reg.csv'))

    m_part1.save(os.path.join(result_dir, 'm_part1_reg.h5'))
    model.save(os.path.join(result_dir, 'model_reg.h5'))


if __name__ == '__main__':
    model_type = 'regression'  # classification
    frag2vec_type = 'mol2vec'   # random / parallel_frag2vec / mol2vec (model name) / tandem_frag2vec
    root_dir = './big-data/moses_dataset/'
    result_dir = None
    mol2md_info_file = './big-data/moses_dataset/result/mol2md_downsampled_max_5000.csv'
    frag2vec_file = os.path.join(root_dir, 'best_model', 'sub_ws_4_minn_1_maxn_2_parallel',
                                 'frag2vec_ws_4_minn_1_maxn_2.csv')
    mol2vec_fp = None
    ref_training_set_mol2vec_fp = None
    ref_test_set_mol2vec_fp = None
    if frag2vec_type == 'random':
        result_dir = './big-data/moses_dataset/random_frag_vec'
        mol2vec_fp = os.path.join(root_dir, 'random_frag_vec', 'random_mol2vec_downsampled.csv')
    elif frag2vec_type == 'mol2vec':
        result_dir = './big-data/moses_dataset/model_mol2vec'
        mol2vec_fp = os.path.join(root_dir, 'model_mol2vec', 'model_mol2vec_mol2vec_trained_by_all_MOSES.csv')
    elif frag2vec_type == 'tandem':
        result_dir = './big-data/moses_dataset/nn/tandem'
        mol2vec_fp = os.path.join(root_dir, 'result', 'step4_model_{}_mol2vec_downsampled.csv'.format(frag2vec_type))

    cid2frag_fp = os.path.join(root_dir, 'result', 'step1_result.txt')
    log_fp = os.path.join(result_dir, 'log.log')
    training_set_mol2vec_fp = os.path.join(result_dir, 'x_training_set_mol2vec.csv')
    test_set_mol2vec_fp = os.path.join(result_dir, 'x_test_set_mol2vec.csv')

    print('  > Start to split data set...')
    if (not os.path.exists(training_set_mol2vec_fp)) and (not mol2vec_fp):
        sample_result = sampling_train_set(mol2md_info_file)
        get_mol_vec(frag2vec_fp=frag2vec_file, data_set_fp=cid2frag_fp,
                    result_path=training_set_mol2vec_fp, log_fp=log_fp,
                    sub_cid_list=list2dic(sample_result['train_set'].index.to_list()))
        get_mol_vec(frag2vec_fp=frag2vec_file, data_set_fp=cid2frag_fp,
                    result_path=test_set_mol2vec_fp, log_fp=log_fp,
                    sub_cid_list=list2dic(sample_result['test_set'].index.to_list()))

    if os.path.exists(mol2vec_fp):
        ref_training_set_mol2vec_fp = os.path.join(root_dir, 'nn', 'parallel', 'x_training_set_mol2vec.csv')
        ref_test_set_mol2vec_fp = os.path.join(root_dir, 'nn', 'parallel', 'x_test_set_mol2vec.csv')

        mol2vec = pd.read_csv(mol2vec_fp, index_col=0, header=None)
        ref_test_set_mol2vec = pd.read_csv(ref_test_set_mol2vec_fp, index_col=0, header=None)
        ref_training_set_mol2vec = pd.read_csv(ref_training_set_mol2vec_fp, index_col=0, header=None)
        test_set_mol2vec = mol2vec[mol2vec.index.isin(ref_test_set_mol2vec.index)].copy()
        training_set_mol2vec = mol2vec[mol2vec.index.isin(ref_training_set_mol2vec.index)].copy()
        test_set_mol2vec.to_csv(test_set_mol2vec_fp, header=False, float_format='%.3f')
        training_set_mol2vec.to_csv(training_set_mol2vec_fp, header=False, float_format='%.3f')

    # deal with y
    print('  > Start to deal with y...')
    mol2md_info = pd.read_csv(mol2md_info_file, index_col='cid')
    if model_type == 'classification':
        mol2md_info[mol2md_info >= 1] = 1
    x_train = pd.read_csv(training_set_mol2vec_fp, header=None, index_col=0)
    x_test = pd.read_csv(test_set_mol2vec_fp, header=None, index_col=0)
    y_train = mol2md_info.loc[x_train.index, :].copy()
    y_test = mol2md_info.loc[x_test.index, :].copy()
    y_test_fp = os.path.join(result_dir, 'y_test_{}.csv'.format(model_type))
    y_train_fp = os.path.join(result_dir, 'y_train_{}.csv'.format(model_type))
    if not os.path.exists(y_test_fp):
        y_test.to_csv(y_test_fp)
    if not os.path.exists(y_train_fp):
        y_train.to_csv(y_train_fp)

    # train model
    print('  > Start to train model...')
    nn_model_regression(x=x_train, y=y_train, result_dir=result_dir)

