"""
helper function for molFrag2vec
"""

import datetime
import os
import re
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from pub_func import get_mol, MAIN_ELEMENT, if_only_main_element, sanitize, get_smiles
from sklearn.model_selection import train_test_split
import fasttext
import csv
import json
import networkx as nx
from mordred import Calculator, descriptors
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


ELEMENTS = ['S', 'Br', 'O', 'C', 'F', 'P', 'N', 'I', 'Cl', 'H']
BONDS = ['DOUBLE', 'SINGLE', 'TRIPLE']
SELECTED_MD = ['nN', 'nS', 'nO', 'nX', 'nBondsD', 'nBondsT', 'naRing', 'nARing']
PRMIER_NUM = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


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


def get_mol_vec(frag2vec_fp, data_set_fp, result_path,
                log_fp, data_set_format='csv', sub_cid_list=None):
    """
    sum all fragment vector to get molecule vector
    :param frag2vec_fp: file path of fragment to vector (separated by ",")
    :param data_set_fp: file path of step1_result.txt or step2_id2frag_info_refrag.csv
    :param result_path: the file path of result
    :param log_fp: the file path of log
    :param data_set_format: csv/ txt
    :param sub_cid_list: a selected cid list to filter whole data set
    :return:
    """
    frag2vec_df = pd.read_csv(frag2vec_fp, index_col=0)
    cid2vec = {}
    counter = 0
    with open(data_set_fp, 'r') as handle:
        if data_set_format == 'csv':
            train_set_reader = csv.reader(handle, delimiter='\t')
            for row in train_set_reader:
                calc_this_line = True
                if row[0] != 'cid':
                    # print(row)
                    cid, smiles, frag_id2smiles, frag_id2neighbors, frag_id2mol_inx = row
                    if sub_cid_list and (cid not in sub_cid_list):
                        calc_this_line = False
                    if calc_this_line:
                        frags = list(json.loads(frag_id2smiles).values())
                        # print(frags)
                        try:
                            cid2vec[cid] = frag2vec_df.loc[frags, :].sum().values
                        except KeyError:
                            _frags = []
                            short_frag = []
                            for frag in frags:
                                if frag not in frag2vec_df.index:
                                    print('frag {} is not in frag2vec and removed'.format(frag))
                                    short_frag.append(frag)
                                else:
                                    _frags.append(frag)
                            cid2vec[cid] = frag2vec_df.loc[_frags, :].sum().values
                            with open(log_fp, 'a') as f_handle3:
                                f_handle3.write('Short fragment {} in this molecule:'.format(','.join(short_frag))
                                                + '\t' + '\t'.join(row) + '\n')
                            # print('fragments {} are not in lib'.format(frags))
                        if len(cid2vec) == 100000:
                            pid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
                            pid2vec_df.to_csv(result_path, mode='a', header=False, float_format='%.3f')
                            cid2vec = {}
                if counter % 10000 == 0:
                    print('>>> Processing line {}...'.format(counter))
                counter += 1
        else:
            for row in handle:
                cid, frag_id = row.strip().split('\t')
                frags = frag_id.split(',')
                try:
                    cid2vec[cid] = frag2vec_df.loc[frags, :].sum().values
                except KeyError:
                    print('some fragments in {} are not in lib'.format(row))
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


def count_fragment(cid2frag_fp):
    """
    count fragment in all training set
    :param cid2frag_fp: cid2fragment file path, i.e. step2_result file
    :return:
    """
    frag2num = {}
    with open(cid2frag_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            cid, sentence = i.strip().split('\t')
            frags = sentence.split(',')
            for frag in frags:
                if frag not in frag2num:
                    frag2num[frag] = 0
                frag2num[frag] += 1
            counter += 1
    frag2num_df = pd.DataFrame.from_dict(frag2num, orient='index')
    frag2num_df.sort_values(by=0, inplace=True, ascending=False)
    frag2num_df.reset_index(inplace=True)
    frag2num_df.rename(columns={0: 'count', 'index': 'fragment'}, inplace=True)
    return frag2num_df


def count_fragment_from_step1(step1_result_fp):
    """
    count fragment in all training set
    :param step1_result_fp: cid2fragment file path, i.e. step2_result file
    :return:
    """
    frag2num = {}

    with open(step1_result_fp, 'r') as input_f:
        counter = 0
        for row in input_f:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            row = row.strip().split('\t')
            if row[0] == 'cid':
                current_row = row
            else:
                current_row = [json.loads(i) for i in row]
            try:
                cid = current_row[0]
            except KeyError:
                cid = ''
            if row[0] != 'cid':  # remove the first line
                id2smiles = {int(i): j for i, j in current_row[2].items()}  # frag_id2smiles
                for k,v in id2smiles.items():
                    if v not in frag2num:
                        frag2num[v] = 0
                    frag2num[v] += 1
            counter += 1
    frag2num_df = pd.DataFrame.from_dict(frag2num, orient='index')
    frag2num_df.sort_values(by=0, inplace=True, ascending=False)
    frag2num_df.reset_index(inplace=True)
    frag2num_df.rename(columns={0: 'count', 'index': 'fragment'}, inplace=True)
    frag2num_df['frequency'] = frag2num_df['count'] / frag2num_df['count'].sum()
    return frag2num_df


def get_fragment_sentence(frag2num_fp, cid2frag_fp, result_fp, result_fp2, replace_by_id=True):
    """
    get fragment sentence
    replace SMILES of fragment by fragment id for saving storage space
    :param frag2num_fp: file path of fragment to number, frag_id,fragment(SMILES),count
    :param cid2frag_fp: file path of cid to sentence (fragments)
    :param result_fp: file path of cid to fragment id
    :param result_fp2: file path of fragment id sentence (separated by space)
    :return:
    """
    frag2num = pd.read_csv(frag2num_fp, index_col='fragment')
    frag2id = frag2num['frag_id'].to_dict()  # a dict of mapping fragment SMILES to fragment id

    with open(cid2frag_fp, 'r') as f_handle:
        counter = 0
        for i in f_handle:
            if counter % 500000 == 0:
                t = datetime.datetime.now()
                print('>>> Current line: {}'.format(counter), t.strftime("%c"))
            cid, sentence = i.strip().split('\t')
            frags = sentence.split(',')
            if replace_by_id:
                frags_id = [str(frag2id[f]) for f in frags]
                with open(result_fp, 'a') as f_handle2:
                    f_handle2.write(cid + '\t' + ','.join(frags_id) + '\n')
            else:
                frags_id = frags
                result_fp2 = result_fp2.replace('frag_id', 'frag_smiles')
            with open(result_fp2, 'a') as f_handle3:
                f_handle3.write(' '.join(frags_id) + '\n')
            counter += 1


def get_sentence_by_cid(cid2sentence, cid_list, result_file, result_file2):
    """
    get cid2smiles.txt in training set
    :param cid2sentence: file path, cid, sentence. step2_result
    :param cid_list： file path of cid list in train_set
    :param result_file: file path of result, cid2sentence which has same order as cid_list
    :param result_file2: only sentence (fragment id)
    :return:
    """
    cid2frag_id_dict = {}
    with open(cid2sentence, 'r') as f_handle:
        for i in f_handle:
            i = i.strip()
            cid, frag_ids = i.split('\t')
            cid2frag_id_dict[cid] = frag_ids
    with open(cid_list, 'r') as f_handle2:
        for i in f_handle2:
            i = i.strip()
            cid = i.split('\t')[0]
            if cid in cid2frag_id_dict:
                with open(result_file, 'a') as r_handle:
                    r_handle.write(cid + '\t' + cid2frag_id_dict[cid] + '\n')
                with open(result_file2, 'a') as r_handle2:
                    r_handle2.write(' '.join(cid2frag_id_dict[cid].split(',')) + '\n')
            else:
                print('>>> this compound {} does not exist in our cid2smiles.txt list...'.format(cid))


class Refragment(object):
    def __init__(self, g, f2f, smiles, test=False):
        """
        g: moelcular graph created by networkx with SMILES and mol_inx for each node
        f2f: fragment2frequency, a dataframe with fragment(SMILES) as index, count/frequency, same as f2n
        smiles: the SMILES of the whold molecule
        mol_inx means the index of each atom in the whole molecule, it's a unique id for each atom
        """
        self.g = g
        self.smiles = smiles
        self.f2f = f2f
        self.test = test

    def get_node_by_degree(self, d=1):
        """
        # The node degree is the number of edges adjacent to the node
        degree equals to 1 means all leaves on the end of graph
        :param d: degree, 1/2
        :return: return all nodes with specific degree
        """
        node_degree = dict(self.g.degree)
        return [k for k, v in node_degree.items() if v == d]

    def get_degree_by_node(self, node_id):
        node_degree = dict(self.g.degree)
        return node_degree[node_id]

    def get_neighbors(self, node_id):
        neigs = list(self.g.neighbors(node_id))
        return neigs

    def check_if_merge(self, node1_id, node2_id):
        """
        check if need to merge these two nodes depend on the frequency of each node
        """
        mean_freq = self.get_mean_frequency()
        node1_freq = self.get_freq(node1_id)
        node2_freq = self.get_freq(node2_id)
        if self.test:
            print('  >type of each element:', type(mean_freq), type(node1_freq), type(node2_freq))
        if (node1_freq >= mean_freq) and (node2_freq >= mean_freq):
            return True
        return False

    def get_freq(self, node_id):
        """
        get frequency by node id
        """
        smiles = self.get_node_attr(node_id, 'smiles')
        # print('current fragment SMILES is: {}'.format(smiles))
        # print(self.f2f.loc[smiles])
        return self.f2f.loc[smiles, 'frequency']

    def get_node_attr(self, node_id, attr):
        """
        get node attribute by node id
        attr: smiles/mol_inx
        """
        if self.test:
            print(node_id)
            print(type(self.g.nodes[node_id]))
        return self.g.nodes[node_id].get(attr, '')

    # def set_node_attr()

    def get_mean_frequency(self, min_count=3):
        """
        mean of the frequency for all fragments which count >= min_count
        """
        mean_freq = self.f2f.loc[self.f2f['count'] >= min_count, 'frequency'].mean()
        return mean_freq

    def _merge_smiles(self, node1_id, node2_id):
        node1_inx_cluster = self.get_node_attr(node1_id, 'mol_inx')
        node2_inx_cluster = self.get_node_attr(node2_id, 'mol_inx')
        if self.test:
            print('  >The mol_inx of node {} is {}'.format(node1_id, node1_inx_cluster))
            print('  >The mol_inx of node {} is {}'.format(node2_id, node2_inx_cluster))
        inx_cluster = set(node1_inx_cluster) | set(node2_inx_cluster)
        merged_smiles = self._get_smiles_by_inx(inx_cluster)
        return {'merged_smiles': merged_smiles, 'merged_inx': inx_cluster}

    def merge_two_nodes(self, left_id, right_id):
        """
        remove right node to left node, and right_id will be delete;
        merge SMILES of these two nodes;
        add new fragment to self.f2f;
        update count and frequency in self.f2f
        """
        if self.check_if_merge(left_id, right_id):
            raw_smiles_left = self.g.nodes[left_id]['smiles']
            raw_smiles_right = self.g.nodes[right_id]['smiles']
            g2 = nx.contracted_nodes(self.g, left_id, right_id, self_loops=False)
            merged_result = self._merge_smiles(left_id, right_id)
            merged_smiles = merged_result['merged_smiles']
            g2.nodes[left_id]['smiles'] = merged_smiles
            g2.nodes[left_id]['mol_inx'] = list(merged_result['merged_inx'])
            if self.test:
                print('  >Merged result is: {}'.format(merged_result))
                print('  >New network: {}'.format(g2.nodes(data=True)))

            if merged_smiles not in self.f2f.index:
                self.f2f.loc[merged_smiles, 'count'] = 0
            self.f2f.loc[merged_smiles, 'count'] += 1
            self.f2f.loc[raw_smiles_left, 'count'] -= 1
            self.f2f.loc[raw_smiles_right, 'count'] -= 1
            self.f2f['frequency'] = self.f2f['count'] / self.f2f['count'].sum()
            self.g = g2.copy()

    def _get_mol(self):
        """
        SMILES -> mol obj
        :param smiles:
        :return:
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            return None
        Chem.Kekulize(mol)
        return mol

    def _get_smiles_by_inx(self, inx_cluster):
        """
        get a subset smiles in the whole molecule by inx_cluster
        :param inx_cluster: a set of atom index in molecule, at least contains two elements
        :return:
        """
        mol = self._get_mol()
        if self.test:
            print('  >atom index cluster: {}'.format(inx_cluster))
        smiles = Chem.MolFragmentToSmiles(mol, inx_cluster, kekuleSmiles=True)
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # new_mol = copy_edit_mol(new_mol).GetMol()
        new_mol = sanitize(new_mol)  # We assume this is not None
        return get_smiles(new_mol)

    def update(self):
        """
        main part of this class
        find all leaves (only have one neighbor) and merge with their neighbor if needed
        """
        for d in range(1,3):
            # d is 1 or 2
            if self.test:
                print('---------------------------------degree {}--------------------------'.format(d))
            nodes = self.get_node_by_degree(d=d)  # a list of node id
            for node in nodes:
                if node in list(self.g.nodes):
                    neighbors = self.get_neighbors(node)  # a list of node id
                    if self.test:
                        print()
                        print('## Current node is: {}'.format(node))
                        print('  >>> Neighbors of this node are : {}'.format(','.join([str(i) for i in neighbors])))
                    for neighbor in neighbors:
                        # neighbor may be deleted on this process, so need to check if it exists
                        if d == 1:  # degree = 1, only leaves
                            if self.test:
                                print('  >>> Start to check if {} and {} can be merged...'.format(neighbor, node))
                            if (neighbor in list(self.g.nodes)) and self.check_if_merge(neighbor, node):
                                if self.test:
                                    print('  >>> Start to merge {} to {}...'.format(node, neighbor))
                                self.merge_two_nodes(left_id=neighbor, right_id=node)
                        if d == 2:  # degree = 2, only merge with the neighbor which degree is 2
                            if self.get_degree_by_node(neighbor) == 2:
                                if self.test:
                                    print('    >the degree of neighbor {} is {}'.format(neighbor, self.get_degree_by_node(neighbor)))
                                    print('  >>> Start to check if {} and {} can be merged...'.format(neighbor, node))
                                if (neighbor in list(self.g.nodes)) and self.check_if_merge(neighbor, node):
                                    if self.test:
                                        print('  >>> Start to merge {} to {}...'.format(neighbor, node))
                                    self.merge_two_nodes(left_id=node, right_id=neighbor)

        n2n = {n: list(self.g.neighbors(n)) for n in list(self.g.nodes())}  # node 2 neighbors, {id: [], ... }
        id2smiles = nx.get_node_attributes(self.g, 'smiles')
        id2mol_inx = nx.get_node_attributes(self.g, 'mol_inx')
        return {'n2n': n2n, 'id2smiles': id2smiles, 'f2f': self.f2f, 'id2mol_inx': id2mol_inx}


def cal_md_by_smiles(smiles_list, md_list=None):
    """
    calculate molecular descriptors by Mordred, https://github.com/mordred-descriptor/mordred
    :param smiles_list: a list of smiles
    :param md_list: a list of MD that need to output
    :return:
    """
    print('  >There are {} SMILES in this list'.format(len(smiles_list)))
    calc = Calculator([descriptors.AtomCount, descriptors.BondCount, descriptors.RingCount], ignore_3D=True)
    mols = []
    for smiles in smiles_list:
        mols.append(Chem.MolFromSmiles(smiles))
    md_df = calc.pandas(mols)
    if not md_list:
        # naRing means aromatic ring count, nARing means aliphatic ring count
        md_list = SELECTED_MD
    md_df['smiles'] = smiles_list
    md_df = md_df.loc[:, ['smiles'] + md_list].copy()
    print('  >The shape of smiles_info is: {}'.format(md_df.shape))
    md_df.rename(columns={'smiles': 'fragment'}, inplace=True)
    md_df.set_index('fragment', inplace=True)
    return md_df


def mul_list(a_list):
    result = 1
    for i in a_list:
        result *= i
    return result


def get_class(frag_info, selected_md=None, min_number=3):
    """
    get unique class depends on different molecular descriptors
    frag_info: a dataframe which contains fragment smiles, selected_md
    selected_md: selected molecular descriptors
    min_number: the minimal number of fragment in each class
    :return: fragment, class(the product of multiple primer numbers), class_id(0 to n), class_num(count each class)
    """
    if not selected_md:
        selected_md = SELECTED_MD
    md_num = len(selected_md)
    if md_num <= len(PRMIER_NUM):
        unique_code = PRMIER_NUM[:md_num]
    else:
        raise Exception('Please give more primer number to PRMIER_NUM...')
    # frag_info = frag_info.set_index('fragment')
    frag_info = frag_info.loc[:, selected_md].copy()
    frag_info[frag_info >= 1] = 1
    frag_info = frag_info.apply(lambda x: np.multiply(x, unique_code), axis=1)
    frag_info[frag_info == 0] = 1
    frag2class = frag_info.apply(lambda x: mul_list(x), axis=1)
    frag2class = pd.DataFrame(frag2class, columns=['class'])

    frag_class2num = {}
    for c in frag2class.index:
        class_num = frag2class.loc[c, 'class']
        if class_num not in frag_class2num:
            frag_class2num[class_num] = 0
        frag_class2num[class_num] += 1
    frag_class2num_df = pd.DataFrame.from_dict(frag_class2num, orient='index', columns=['class_num'])
    frag2class = frag2class.merge(frag_class2num_df, left_on='class', right_index=True)
    frag2class = frag2class[frag2class['class_num'] >= min_number].copy()
    print('  >the shape of frag2class after filtered: {}'.format(frag2class.shape))

    unique_class = sorted(frag2class['class'].unique())
    code2id = {unique_class[i]: i for i in range(len(unique_class))}
    print(code2id)
    frag2class['class_id'] = frag2class['class'].apply(lambda x: code2id[x])

    # depth = len(code2id)
    # y_one_hot = tf.one_hot(frag2class_filtered.class_id.values, depth=depth)
    # print('  >the shape of one hot y: {}'.format(y_one_hot.shape))
    return frag2class


def get_class_md_combination(frag_info, selected_md=None, min_number=3):
    """
    get unique class depends on different molecular descriptors
    frag_info: a dataframe which contains fragment smiles, selected_md
    selected_md: selected molecular descriptors
    min_number: the minimal number of fragment in each class
    :return: fragment, class(the combination of different MD, such as 10001010),
             class_id(0 to n), class_num(count each class)
    """
    if not selected_md:
        selected_md = SELECTED_MD
    # md_num = len(selected_md)
    # if md_num <= len(PRMIER_NUM):
    #     unique_code = PRMIER_NUM[:md_num]
    # else:
    #     raise Exception('Please give more primer number to PRMIER_NUM...')
    # frag_info = frag_info.set_index('fragment')
    frag_info = frag_info.loc[:, selected_md].copy()
    frag_info[frag_info >= 1] = 1
    # frag_info = frag_info.apply(lambda x: np.multiply(x, unique_code), axis=1)
    # frag_info[frag_info == 0] = 1
    frag2class = frag_info.apply(lambda x: ''.join([str(i) for i in x]), axis=1)
    frag2class = pd.DataFrame(frag2class, columns=['class'])

    frag_class2num = {}
    for c in frag2class.index:
        class_num = frag2class.loc[c, 'class']
        if class_num not in frag_class2num:
            frag_class2num[class_num] = 0
        frag_class2num[class_num] += 1
    frag_class2num_df = pd.DataFrame.from_dict(frag_class2num, orient='index', columns=['class_num'])
    frag2class = frag2class.merge(frag_class2num_df, left_on='class', right_index=True)
    frag2class = frag2class[frag2class['class_num'] >= min_number].copy()
    print('  >the shape of frag2class after filtered: {}'.format(frag2class.shape))

    unique_class = sorted(frag2class['class'].unique())
    code2id = {unique_class[i]: i for i in range(len(unique_class))}
    print(code2id)
    frag2class['class_id'] = frag2class['class'].apply(lambda x: code2id[x])

    # depth = len(code2id)
    # y_one_hot = tf.one_hot(frag2class_filtered.class_id.values, depth=depth)
    # print('  >the shape of one hot y: {}'.format(y_one_hot.shape))
    return frag2class


def get_xy(frag2vec, frag2class, d=100):
    """
    merge frag2vec and frag_info, then get vector and one-hot y
    :param: d, dimension of fragment vector
    """
    # frag2vec = pd.read_csv(frag2vec_fp, index_col='frag_id')
    # frag_info = pd.read_csv(frag_info_fp, index_col='fragment')
    frag2class = frag2class.loc[:, ['class']].copy()
    class_id = np.unique(frag2class.values)
    print('there are {} unique classes: {}'.format(len(np.unique(frag2class.values)), class_id))
    class2inx = {class_id[i]: i for i in range(len(class_id))}
    print(class2inx)
    frag2vec = frag2vec.merge(frag2class, left_index=True, right_index=True)
    depth = len(class_id)
    y_inx = frag2vec.loc[:, ['class']].apply(lambda x: class2inx[x.values[0]], axis=1)
    # print(y_inx, type(y_inx))
    y = tf.one_hot(y_inx.values, depth=depth)
    print('  >the first 2 classis {}'.format(frag2vec.loc[:, ['class']].head(2)))
    return {'x': frag2vec.iloc[:, range(d)], 'y': y}


def vis_class(X, labels, title, file_path=None):
    """
    plot reduced fragment vector
    :param X: two dimensions np.array
    :param labels:
    :param title:
    :param file_path:
    :return:
    """
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]

    plt.figure(figsize=(15, 12))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14, label=k)
        plt.text(xy[0, 0], xy[0, 1], str(k), fontsize=18)

    #         xy = X[class_member_mask & ~core_samples_mask]
    #         plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
    #                  markeredgecolor='k', markersize=6, label=k)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path, dpi=300)


class SuperviseClassModel(object):
    def __init__(self, n_output):
        self.n_output = n_output
        self.m_part1 = keras.Sequential([keras.layers.Dense(50, activation='tanh', input_shape=[100]),
                                         keras.layers.Dense(30, activation='relu'),
                                         keras.layers.Dense(50, activation='tanh'),
                                         keras.layers.Dense(100, activation='relu')])
        self.m_part2 = keras.Sequential([keras.layers.Dense(self.n_output, input_shape=[100])])
        self.model = keras.Sequential([self.m_part1, self.m_part2])

    def model_compile(self):
        self.model.compile(optimizer='rmsprop',
                           loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def training(self, x, y):
        self.model.fit(x, y, epochs=50, batch_size=32)

    def get_embedding_vec(self, x):
        prob_m_part1 = tf.keras.Sequential([self.m_part1, tf.keras.layers.Softmax()])
        return prob_m_part1.predict(x)


def tmp_bug_fix(a_str):
    """
    split a str by 'id+number'
    :param a_str:
    :return:
    """
    result_str = a_str
    if re.search(r'id\d+', a_str):
        splited_str = re.split(r'id', a_str)
        splited_str = ['id'+i for i in splited_str if i != '']
        result_str = '\n'.join(splited_str)
    return result_str


if __name__ == '__main__':
    # replace separator
    root_dir = r'/home/belter/github/my-research/fragTandem2vec/big-data/moses_dataset'
    # step1_result_file_names = 'step1_result.txt'
    # step1_result_fp = os.path.join(root_dir, 'result', step1_result_file_names)
    # frag2num = count_fragment_from_step1(step1_result_fp=step1_result_fp)
    # frag2num.to_csv(os.path.join(root_dir, 'result', 'step1_frag2num.csv'), index=False)
    step2_result_file_name = 'step2_parallel_refrag_cid2smiles_sentence.csv'
    step2_result_fp = os.path.join(os.path.join(root_dir, 'result', step2_result_file_name))
    with open(step2_result_fp, 'r') as f_handle:
        for i in f_handle:
            i = i.strip()
            i = tmp_bug_fix(i)
            with open(step2_result_fp.replace('.csv', '_fixed.csv'), 'a') as f_handle2:
                f_handle2.write(i + '\n')

    # # get fragment attribution
    # frag2vec = pd.read_csv(os.path.join(root_dir, 'frag2vec.csv'), index_col=0)
    # # atom_type = []
    # # bond_type = []
    # cols = ['aromatic_ring', 'n_atom', 'ring_num'] + ELEMENTS + BONDS
    # frag_attr_df = pd.DataFrame(columns=cols,
    #                             index=frag2vec.index)
    # # frag2attr = {}
    # for i in frag2vec.index:
    #     frag_attr = get_frag_attr(i)
    #     atom_num = [frag_attr['atom2num'][i] for i in ELEMENTS]
    #     bond_num = [frag_attr['bond2num'][i] for i in BONDS]
    #     val = [frag_attr['aromatic_ring'], frag_attr['n_atom'], frag_attr['ring_num']] + atom_num + bond_num
    #     frag_attr_df.loc[i, :] = val
    #     # frag2attr[i] = val
    # frag_attr_df.to_csv(os.path.join(root_dir, 'frag_attr.csv'))

