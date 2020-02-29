"""
added to helper_func.py
"""
import os
import numpy as np
import pandas as pd
import csv


def get_mol_vec(frag2vec, data_set, result_path):
    """
    sum all fragment vector to get molecule vector
    :param frag2vec:
    :param data_set: step5_x_training_set.csv
    :return:
    """
    frag2vec_df = pd.read_csv(frag2vec, index_col=0)
    cid2vec = {}
    counter = 0
    with open(data_set, 'r') as handle:
        train_set_reader = csv.reader(handle, delimiter=',')
        for row in train_set_reader:
            if row[-1] != '0':
                cid, mol_path, mol_inx, frag_smiles = row
                frags = frag_smiles.split(' ')
                try:
                    cid2vec[cid] = frag2vec_df.loc[frags, :].sum().values
                except KeyError:
                    print('fragments {} are not in lib'.format(frag_smiles))
                if len(cid2vec) == 500000:
                    pid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
                    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')
                    cid2vec = {}
            if counter % 10000 == 0:
                print('>>> Processing line {}...'.format(counter))
            counter += 1
    # the last part
    pid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')


if __name__ == '__main__':
    root_dir = './big-data'
    frag2vec_file = os.path.join(root_dir, 'frag2vec_molFrag2vec.csv')
    training_set_file = os.path.join(root_dir, 'cid2fragment', 'molFrag2vec', 'selected_cid2fragment.csv')
    result_file = os.path.join(root_dir, 'vectors', 'mol2vec_model_molFrag2vec.csv')
    get_mol_vec(frag2vec=frag2vec_file, data_set=training_set_file, result_path=result_file)
