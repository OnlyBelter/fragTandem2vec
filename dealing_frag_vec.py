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
    pid2vec = {}
    counter = 0
    with open(data_set, 'r') as handle:
        train_set_reader = csv.reader(handle, delimiter='\t')
        for row in train_set_reader:
            if row[0] != '0':
                pid, mol_path, mol_inx, frag_smiles = row
                frags = frag_smiles.split(',')
                try:
                    pid2vec[pid] = frag2vec_df.loc[frags, :].sum().values
                except KeyError:
                    print('fragments {} are not in lib'.format(frag_smiles))
                if len(pid2vec) == 500000:
                    pid2vec_df = pd.DataFrame.from_dict(pid2vec, orient='index')
                    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')
                    pid2vec = {}
            if counter % 100000 == 0:
                print('>>> Processing line {}...'.format(counter))
            counter += 1
    # the last part
    pid2vec_df = pd.DataFrame.from_dict(pid2vec, orient='index')
    pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')


if __name__ == '__main__':
    root_dir = '/home/belter/github/my-research/jtnn-py3/big-data'
    frag2vec_file = os.path.join(root_dir, 'process', 'frag2vec.csv')
    training_set_file = os.path.join(root_dir, 'process', 'step5_x_training_set.csv')
    result_file = os.path.join(root_dir, 'mol2vec.csv')
    get_mol_vec(frag2vec=frag2vec_file, data_set=training_set_file, result_path=result_file)
