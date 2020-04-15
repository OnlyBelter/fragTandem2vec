"""
cid -> smiles -> fragment -> frag vector -> mol vector
step4: merge fragment vectors to a molecular vector
"""
import os
import numpy as np
import pandas as pd
from helper_func import get_mol_vec
# from mol2vec_related.helper_func import load_trained_model
# from mol2vec.features import MolSentence, DfVec, sentences2vec


def get_fragment_by_cid(selected_cid_fp, all_train_cid_fp, fragment_fp, result_fp):
    """
    get fragment of selected cid
    :param selected_cid_fp: selected cids which have class from classyFire
    :param all_train_cid_fp: all training set, contain cid/smiles (sentence, fragments)
    :param fragment_fp: all fragments (sentence) of all training set, same order as all_train_cid file
    :param result_fp:
    :return:
    """
    selecred_cid = pd.read_csv(selected_cid_fp)
    try:
        inx2cid = selecred_cid['CID'].to_dict()  # index in selected_cid file
    except KeyError:
        inx2cid = selecred_cid['0'].to_dict()
    # print(inx2cid)
    cid2inx = {j: i for i,j in inx2cid.items()}
    cid2order = {}  # order in all_train_cid file
    print('>>> start to generate cid2order...', '\n')
    with open(all_train_cid_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            i = i.strip().split('\t')
            cid = int(i[0])
            if cid in cid2inx:
                cid2order[cid] = counter
            counter += 1
    # print('cid2order: ', cid2order)
    order2cid = {j: i for i,j in cid2order.items()}
    order2fragment = {}  # order in fragment file, same as all_train_cid file
    print('>>> start to generate order2fragment...', '\n')
    with open(fragment_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            i = i.strip()
            if counter in order2cid:
                order2fragment[counter] = i
            counter += 1
    cid2order_df = pd.DataFrame.from_dict(cid2order, orient='index')
    cid2order_df.rename(columns={0: 'order'}, inplace=True)
    print(cid2order_df.head())
    cid2inx_df = pd.DataFrame.from_dict(cid2inx, orient='index')
    cid2inx_df.rename(columns={0: 'inx'}, inplace=True)
    print(cid2inx_df.head())
    cid2inx_with_order = cid2inx_df.merge(cid2order_df, left_index=True, right_index=True)
    print(cid2inx_with_order.head())
    order2fragment_df = pd.DataFrame.from_dict(order2fragment, orient='index')
    order2fragment_df.rename(columns={0: 'fragment'})
    inx2fragment = cid2inx_with_order.merge(order2fragment_df, left_on='order', right_index=True)
    inx2fragment.sort_values(by=['inx'], inplace=True)
    inx2fragment.to_csv(result_fp, index_label='cid')


if __name__ == '__main__':
    frag2vec_type = 'parallel'   # random / parallel_frag2vec / tandem_frag2vec
    root_dir = 'big-data/moses_dataset/'
    result_dir = os.path.join(root_dir, 'result')
    # result_dir = 'big-data/moses_dataset/result'
    # download_big_data_dir = './big-data'
    # include_small_dataset_dir = './dataset'
    # result_fp = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_mol2vec.csv')
    # result_fp2 = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_fragTandem2vec.csv')
    selected_cid_fp = os.path.join(result_dir, 'mol2md_downsampled_max_5000.csv')
    cid2frag_fp = os.path.join(result_dir, 'step1_result.txt')
    # frag2vec_file = os.path.join(root_dir, 'best_model', 'sub_ws_4_minn_1_maxn_2_parallel',
    #                                        'frag2vec_ws_4_minn_1_maxn_2.csv')
    frag2vec_file = os.path.join(root_dir, 'best_model', 'sub_ws_10_minn_1_maxn_2_tandem',
                                 'frag2vec_ws_10_minn_1_maxn_2.csv')
    result_file = os.path.join(result_dir, 'step4_model_{}_mol2vec_downsampled.csv'.format(frag2vec_type))
    log_fp = os.path.join(result_dir, 'step4_log.log')
    if frag2vec_type == 'random':
        # np.random.seed()
        frag2vec = pd.read_csv(frag2vec_file, index_col='fragment')
        frag2vec_file = os.path.join(root_dir, 'random_frag_vec', 'frag2vec_random.csv')
        result_file = os.path.join(root_dir, 'random_frag_vec', 'random_mol2vec_downsampled.csv')
        log_fp = os.path.join(root_dir, 'random_frag_vec', 'log.log')
        if not os.path.exists(frag2vec_file):
            frag2vec_random = pd.DataFrame(data=np.random.random(size=frag2vec.shape), index=frag2vec.index)
            frag2vec_random.to_csv(frag2vec_file)
    # training_set_file = result_fp2

    #
    # # get fragments of selected in molFrag2vec
    # # selected_cid_fp = './big-data/cid2class_classyfire/down_sampled_cid2class_unique.csv'
    # all_train_cid_fp2 = os.path.join(download_big_data_dir, 'cid2fragment', 'molFrag2vec', 'x_training_set_cid2_sentence_new.csv')
    # all_fragment_fp2 = os.path.join(download_big_data_dir, 'cid2fragment', 'molFrag2vec', 'x_training_set_sentence_new.csv')
    # get_fragment_by_cid(selected_cid_fp=selected_cid_fp, all_train_cid_fp=all_train_cid_fp2,
    #                     fragment_fp=all_fragment_fp2, result_fp=result_fp2)

    # get vector of each molecule by molFrag2vec model
    selected_cid = pd.read_csv(selected_cid_fp)
    sub_cid_list = selected_cid['cid'].to_list()
    sub_cid = {cid: 0 for cid in sub_cid_list}
    get_mol_vec(frag2vec_fp=frag2vec_file, data_set_fp=cid2frag_fp,
                result_path=result_file, log_fp=log_fp, sub_cid_list=sub_cid)

