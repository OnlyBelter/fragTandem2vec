"""
cid -> smiles -> fragment -> vector
step1: cid -> SMILES -> fragments, 'helper_func.py'
"""
import pandas as pd
import os
from helper_func import get_mol_vec
from mol2vec_related.helper_func import load_trained_model
from mol2vec.features import MolSentence, DfVec, sentences2vec


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


if __name__=='__main__':
    result_dir = './demo_data'
    download_big_data_dir = './big-data'
    include_small_dataset_dir = './dataset'
    result_fp = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_mol2vec.csv')
    result_fp2 = os.path.join(result_dir, 'step4_selected_cid2fragment_down_sampled_model_fragTandem2vec.csv')
    selected_cid_fp = os.path.join(include_small_dataset_dir, 'down_sampled_cid2class_unique.csv')

    # # get fragments of selected molecule in mol2vec
    # print(os.path.abspath(os.getcwd()))
    # print(os.path.exists(selected_cid_fp))
    # all_train_cid_fp = os.path.join(download_big_data_dir, 'cid2fragment', 'mol2vec', 'cid2smiles_training_set.txt')
    # all_fragment_fp = os.path.join(download_big_data_dir, 'cid2fragment', 'mol2vec', 'cid2smiles_training_set_coupus.txt')
    # get_fragment_by_cid(selected_cid_fp=selected_cid_fp, all_train_cid_fp=all_train_cid_fp,
    #                     fragment_fp=all_fragment_fp, result_fp=result_fp)
    #
    # # get fragments of selected in molFrag2vec
    # # selected_cid_fp = './big-data/cid2class_classyfire/down_sampled_cid2class_unique.csv'
    # all_train_cid_fp2 = os.path.join(download_big_data_dir, 'cid2fragment', 'molFrag2vec', 'x_training_set_cid2_sentence_new.csv')
    # all_fragment_fp2 = os.path.join(download_big_data_dir, 'cid2fragment', 'molFrag2vec', 'x_training_set_sentence_new.csv')
    # get_fragment_by_cid(selected_cid_fp=selected_cid_fp, all_train_cid_fp=all_train_cid_fp2,
    #                     fragment_fp=all_fragment_fp2, result_fp=result_fp2)

    # get vector of each molecule by molFrag2vec model
    # frag2vec_file = os.path.join(include_small_dataset_dir, 'fragTandem2vec_related', 'frag2vec_model_fragTandem2vec_new.csv')
    # training_set_file = result_fp2
    # result_file = os.path.join(result_dir, 'step4_selected_mol2vec_model_fragTandem2vec.csv')
    # get_mol_vec(frag2vec_fp=frag2vec_file, data_set_fp=training_set_file, result_path=result_file)

    # get vector of each molecule by mol2vec model
    mol_info = pd.read_csv(os.path.join(include_small_dataset_dir, 'mol2vec_related',
                                        'selected_cid2fragment_down_sampled_model_mol2vec.csv'), index_col='cid')
    model_fp = os.path.join(include_small_dataset_dir, 'mol2vec_related', 'mol2vec_model.pkl')
    model = load_trained_model(model_fp)
    # print(mol_info.loc[4568802, '0'])
    mol_info['sentence'] = mol_info.apply(lambda x: MolSentence([str(i) for i in x['0'].split(' ')]), axis=1)
    # print(mol_info)
    mol_info['mol2vec_related'] = [DfVec(x) for x in sentences2vec(mol_info['sentence'], model)]
    cid2vec = {}
    for cid in mol_info.index.to_list():
        cid2vec[cid] = list(mol_info.loc[cid, 'mol2vec_related'].vec)
    cid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    print(cid2vec_df.shape)
    result_file2 = os.path.join(result_dir, 'step4_selected_mol2vec_model_mol2vec.csv')
    cid2vec_df.to_csv(result_file2, header=False, float_format='%.3f')