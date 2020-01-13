import json
import os
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from pub_func import get_mol, MAIN_ELEMENT, if_only_main_element
from sklearn.model_selection import train_test_split
import fasttext


def replace_separator(root_dir, step1_file_names):
    for file_name in step1_file_names:
        with open(os.path.join(root_dir, file_name), 'r') as f:
            counter = 0
            for row in f:
                row = row.strip()
                if counter == 0:
                    row = row.replace(',', '\t')
                else:
                    row = row.replace('","', '"\t"')
                    row = row.replace('",{', '"\t{')
                    row = row.replace('},{', '}\t{')
                    # row_list = row.split('\t')
                counter += 1
                with open(os.path.join(root_dir, file_name.replace('.csv', '.txt')), 'a') as result_f:
                    result_f.write(row + '\n')


def merge_filter(step2_file_names, short_mol_file,
                 result_file, other_mol_file, frag2count_file):
    # merge all result of step2 together and filter molecular sentence (only keep fragment >= 3)
    # statistics of fragments
    shortest_frag = 3
    frag2count = {}
    for file_name in step2_file_names:
        print('>>> Dealing with file: {} ...'.format(file_name))
        with open(os.path.join(root_dir, file_name), 'r') as f:
            # counter = 0
            for row in f:
                row = row.strip()
                pid, mol_path, frag_inx, frag_smiles = row.split('\t')
                frags = frag_smiles.split(',')
                num_frag = len(frags)
                if num_frag < shortest_frag:
                    # pass
                    with open(short_mol_file, 'a') as short_f:
                        short_f.write(row + '\n')
                else:
                    contain_dot = np.any(['.' in fr for fr in frags])
                    only_main_element = True
                    for fr in frags:
                        if not if_only_main_element(fr):
                            only_main_element = False
                            break
                    if (not contain_dot) and only_main_element:
                        for fr in frags:
                            if fr not in frag2count.keys():
                                frag2count[fr] = 0
                            frag2count[fr] += 1
                        with open(result_file, 'a') as long_f:
                            long_f.write(row + '\n')
                    else:
                        with open(other_mol_file, 'a') as other_f:
                            other_f.write(row + '\n')

    frag2count_pd = pd.DataFrame.from_dict(frag2count, orient='index')
    frag2count_pd.to_csv(frag2count_file, index_label='fragment')


def filter_by_frag_frequent(long_mol, frag2count,
                            filtered_long_mol_file,
                            rare_frag_mol_file):
    """

    :param long_mol:
    :param frag2count:
    :param filtered_long_mol_file:
    :param rare_frag_mol_file:
    :return:
    """
    # mol_sentence_file = os.path.join(result_dir, 'step4_mol_sentence.txt')
    with open(long_mol, 'r') as handle:
        for l in handle:
            l = l.strip()
            pid, mol_path, mol_tree_inx, _frags = l.split('\t')
            frags = _frags.split(',')
            frags_count = [frag2count[f] for f in frags]
            if np.all([c >= 20 for c in frags_count]):
                with open(filtered_long_mol_file, 'a') as r1_handle:
                    r1_handle.write(l + '\n')
            else:
                with open(rare_frag_mol_file, 'a') as r2_handle:
                    r2_handle.write(l + '\t' + ','.join([str(i) for i in frags_count]) + '\n')


def split_training_test(long_mol_file, test_percent=0.1, result_dir=''):
    long_mol = pd.read_csv(long_mol_file, sep='\t', header=None)
    X_train, X_test = train_test_split(long_mol, test_size=test_percent, random_state=42)
    X_train.to_csv(os.path.join(result_dir, 'step5_x_training_set.csv'), sep='\t', index=False)
    X_test.to_csv(os.path.join(result_dir, 'step5_x_test_set.csv'), sep='\t', index=False)
    x_train_mol_sentence_file = os.path.join(result_dir, 'step5_x_train_mol_sentence.csv')
    x_test_mol_sentence_file = os.path.join(result_dir, 'step5_x_test_mol_sentence.csv')
    with open(x_train_mol_sentence_file, 'w') as x_train_handle:
        for i in X_train.index:
            frags = X_train.loc[i, 3]
            x_train_handle.write(' '.join(frags.split(',')) + '\n')
    with open(x_test_mol_sentence_file, 'w') as x_test_handle:
        for i in X_test.index:
            frags = X_test.loc[i, 3]
            x_test_handle.write(' '.join(frags.split(',')) + '\n')
    # X_train.loc[:, [3]].apply(lambda s: s.split(',')).to_csv(x_train_mol_sentence_file,
    #                                                   header=False, index=False, doublequote=False)
    # X_test.loc[:, [3]].apply(lambda s: s.split(',')).to_csv(os.path.join(result_dir, 'step5_x_test_mol_sentence.csv'),
    #                                                  header=False, index=False, doublequote=False)


def get_nn(model_path):
    model = fasttext.load_model(model_path)
    model.get_nearest_neighbors()


def get_frag_attr(frag_smiles):

    bond2num = {i: 0 for i in BONDS}
    atom2num = {i: 0 for i in ELEMENTS}
    aromatic = False
    try:
        mol = get_mol(frag_smiles)
    except:
        mol = None
    if mol is not None:
        n_atoms = mol.GetNumAtoms()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bond_type = ''
            if mol.GetAtomWithIdx(a1).GetIsAromatic() or mol.GetAtomWithIdx(a2).GetIsAromatic():
                aromatic = True
                if mol.GetAtomWithIdx(a1).GetIsAromatic() and mol.GetAtomWithIdx(a2).GetIsAromatic():
                    pass
                else:
                    bond_type = bond.GetBondType()
            else:
                bond_type = bond.GetBondType()
            if bond_type:
                bond_type = str(bond_type)
                if bond_type not in bond2num:
                    bond2num[bond_type] = 0
                bond2num[bond_type] += 1
        # atom2num = {}
        if mol:
            for atom in mol.GetAtoms():
                current_atom = atom.GetSymbol()
                if current_atom not in atom2num.keys():
                    atom2num[current_atom] = 0
                atom2num[current_atom] += 1

        ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]
        return {'bond2num': bond2num, 'atom2num': atom2num,
                'aromatic_ring': aromatic, 'ring_num': len(ssr), 'n_atom': n_atoms}
    else:
        return {'bond2num': bond2num, 'atom2num': atom2num,
                'aromatic_ring': aromatic, 'ring_num': 0, 'n_atom': 0}


if __name__ == '__main__':
    # replace separator
    root_dir = r'/home/belter/github/my-research/jtnn-py3/big-data/process'
    file_names = ['step1_result_part1.csv', 'step1_result_part2.csv',
                  'step1_result_part3.csv', 'step1_result_part4.csv']

    # step 2
    step2_result_files = ['step2_refragment_result_part1_corrected.csv', 'step2_refragment_result_part2_corrected.csv',
                          'step2_refragment_result_part3_corrected.csv', 'step2_refragment_result_part4_corrected.csv']
    # replace_separator(root_dir, file_names)

    # step 3
    short_file_path = os.path.join(root_dir, 'step3_short_mol.txt')
    result_file_path = os.path.join(root_dir, 'step3_long_mol.txt')
    other_mol_file_path = os.path.join(root_dir, 'step3_other_mol.txt')
    frag2count_file_path = os.path.join(root_dir, 'step3_frag2count.csv')
    # merge_filter(step2_result_files, short_mol_file=short_file_path,
    #              result_file=result_file_path, other_mol_file=other_mol_file_path,
    #              frag2count_file=frag2count_file_path)

    # step 4
    filtered_long_mol_file = os.path.join(root_dir, 'step4_filtered_long_mol.txt')
    rare_frag_mol_file = os.path.join(root_dir, 'step4_rare_frag_mol.txt')
    # frag2count_df = pd.read_csv(frag2count_file_path)
    # frag2count = dict(zip(frag2count_df['fragment'], frag2count_df['0']))
    # filter_by_frag_frequent(result_file_path, frag2count,
    #                         filtered_long_mol_file=filtered_long_mol_file,
    #                         rare_frag_mol_file=rare_frag_mol_file)

    # step 5
    # split_training_test(filtered_long_mol_file, test_percent=0.1, result_dir=root_dir)

    # step 6 get fragment attribution
    ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
    BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
    frag2vec = pd.read_csv(os.path.join(root_dir, 'frag2vec.csv'), index_col=0)
    # atom_type = []
    # bond_type = []
    cols = ['aromatic_ring', 'n_atom', 'ring_num'] + ELEMENTS + BONDS
    frag_attr_df = pd.DataFrame(columns=cols,
                                index=frag2vec.index)
    # frag2attr = {}
    for i in frag2vec.index:
        frag_attr = get_frag_attr(i)
        atom_num = [frag_attr['atom2num'][i] for i in ELEMENTS]
        bond_num = [frag_attr['bond2num'][i] for i in BONDS]
        val = [frag_attr['aromatic_ring'], frag_attr['n_atom'], frag_attr['ring_num']] + atom_num + bond_num
        frag_attr_df.loc[i, :] = val
        # frag2attr[i] = val
    frag_attr_df.to_csv(os.path.join(root_dir, 'frag_attr.csv'))

