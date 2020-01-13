import networkx as nx
import matplotlib.pyplot as plt
import os
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import json
import numpy as np
from pub_func import (Mol2Network, PreMerge)
import argparse


def draw_graph(g, file_dir, file_name):
    """
    draw molecular graph
    :param g: molecular graph
    :param file_dir: where to save figure
    :param file_name: file name
    :return:
    """
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.savefig(os.path.join(file_dir, file_name + '.png'), dpi=300)
    plt.close()


def mol2network(n2n, file_path, name, draw_network=False):
    """

    :param n2n: node to neighbors
    :return: a network
    """
    g = nx.Graph()
    for i in n2n.keys():
        edges = [(i, j) for j in n2n[i]]
        g.add_edges_from(edges)
    if draw_network:
        nx.draw(g, with_labels=True, font_weight='bold')
        plt.savefig(os.path.join(file_path, name + '.png'), dpi=300)
        plt.close()
    return g


def split_long_cluster(cluster_inx, min_len=6):
    """
    split long cluster which is longer than min_len
    :param cluster_inx: a list of index in a single cluster, such as ['6', '15', '4', '3', '14', '8', '16', '10']
    :param min_len: 6
    :return: split small clusters, the length of each small cluster should be shorter than min_len, [[], ..., []]
    """
    len_list = len(cluster_inx)
    if len_list <= min_len:
        return [cluster_inx]
    else:
        if len_list % 2 == 0:
            split_inx = int(len_list / 2)
        else:
            split_inx = int((len_list + 1) / 2)
        list_left = cluster_inx[:split_inx]
        list_right = cluster_inx[split_inx:]
        return split_long_cluster(list_left, min_len) + split_long_cluster(list_right, min_len)


def branch_path_helper(current_node, branch, smiles, id2node):
    """
    a single branch path
    :param current_node: current skeleton node
    :param branch: a list of fragment id in a branch of mol tree
    :param smiles: the SMILES of this whole molecule
    :param id2node:
    :return: {'current_node': merged_skeleton_node, 'branch_node': merged_branch_node]
    """
    assert type(branch) == list
    premerge_obj_skeleton = PreMerge(cluster_type='skeleton', smiles=smiles)
    premerge_obj_branch = PreMerge(cluster_type='branch', smiles=smiles, sk_neighbor=current_node)
    premerge_obj_skeleton.add_node(current_node)
    if len(branch) == 1:
        # only have one fragment on the branch
        branch_node = id2node[branch[0]]
        if not branch_node.ring:
            # non-ring node, merge it with skeleton fragment
            premerge_obj_skeleton.add_node(branch_node)

        else:  # 1 ring branch fragment, separate to a single independent cluster
            premerge_obj_branch.add_node(branch_node)

    # 2 fragments in this branch
    # separate to a single independent cluster
    elif len(branch) >= 2:
        for node_id in branch:
            premerge_obj_branch.add_node(id2node[node_id])
    merged_skeleton_node = premerge_obj_skeleton.merge()
    merged_branch_node = premerge_obj_branch.merge()  # may be None
    return {'current_node': merged_skeleton_node, 'branch_node': merged_branch_node}


def deal_with_branch_path(current_node, branch_path, smiles, id2node):
    """
    merge branch path if the length of branch path equals to 1, separate branch as a single cluster if the length >= 2
    :param current_node: a Node obj
    :param branch_path: [[1], [4, 5]], may have multiple branch path
    :param smiles: the SMILES of this whole molecule
    :param id2node:
    :return: {'current_node': Node, 'branch_node': [Node, Node]}
    """
    num_branch = len(branch_path)
    result = {'branch_node': [], 'current_node': None}
    _result = {}
    if num_branch == 1:  # only one branch
        branch = branch_path[0]
        _result = branch_path_helper(current_node=current_node, branch=branch,
                                     smiles=smiles, id2node=id2node)
        current_node = _result['current_node']
        if _result['branch_node'] is not None:
            result['branch_node'].append(_result['branch_node'])
    elif num_branch >= 2:  # need update current_node
        for branch in branch_path:
            _result = branch_path_helper(current_node=current_node, branch=branch,
                                         smiles=smiles, id2node=id2node)
            current_node = _result['current_node']
            if _result['branch_node'] is not None:
                result['branch_node'].append(_result['branch_node'])
    result['current_node'] = current_node

    return result


class Refragment:
    def __init__(self, mol_path, id2node, smiles, log_path, n2n):
        """
        merge small fragments
        :param mol_path: mol skeleton, [1, 12, 3, 8, 4, 5, 6, 9, 7, 10]
        :param id2node: each fragment id to Node class
        :param smiles: the SMILES of this molecule
        :param log_path: the path of log file
        :param n2n: node2neighbor
        :return:
        """
        self.mol_path = mol_path
        self.id2node = id2node
        self.smiles = smiles
        self.mol = Chem.MolFromSmiles(smiles)
        self.log_path = log_path
        self.n2n = n2n
        self.taken_inx = []

    def get_frag_cluster(self, cluster):
        """
        get short fragment cluster with neighbor fragments
        :param cluster: a list of short fragment id, such as ['1', '12', '3'], among mol skeleton
        :return: the SMILES of each fragment in this cluster
        """
        # alt_smiles has the same meaning as c_x at below
        cluster_info = pd.DataFrame(columns=['node_id', 'smiles',
                                             'num_atom', 'branch_path', 'mol_inx'])
        counter = 0
        for node_id in cluster:
            node_id = int(node_id)
            node = self.id2node[node_id]
            # print('node type: ', node.type, 'neighbors: ', node.neighbors)
            branch_neis = node.branch_path  # [[2, 1]]
            cluster_info.loc[counter, 'node_id'] = node_id
            cluster_info.loc[counter, 'smiles'] = node.smiles

            cluster_info.loc[counter, 'branch_path'] = branch_neis
            cluster_info.loc[counter, 'num_atom'] = node.num_atom
            cluster_info.loc[counter, 'mol_inx'] = node.mol_inx

            counter += 1
        return cluster_info

    def split_by_ring(self):
        mol_path_split = []
        short_frag = []
        for mol_id in self.mol_path:
            if not self.id2node[mol_id].ring:
                mol_path_split.append(str(mol_id))
            else:
                mol_path_split += ['|', str(mol_id), '|']
        # print(mol_path_split)
        tmp_frag = ','.join(mol_path_split).split('|')
        # print(tmp_frag)
        tmp_frag = [i.split(',') for i in tmp_frag if i]
        for tmp in tmp_frag:
            tmp = [i for i in tmp if i]
            if len(tmp) >= 1:  # ring node also included
                short_frag.append(tmp)
        return short_frag

    def get_frag_inx(self, frags):
        """
        :param frags: a list of SMILES
        :return: the index of each atom in this fragment
        """
        all_indices = {}
        for frag in frags:
            if frag not in all_indices:
                # print('  frag: ', frag)
                inx = self.mol.GetSubstructMatches(Chem.MolFromSmarts(frag))
                if not inx:
                    try:
                        inx = self.mol.GetSubstructMatches(Chem.MolFromSmiles(frag))
                    except:
                        pass
                all_indices[frag] = inx

        return all_indices

    def merge_frags(self, frag_cluster):
        """
        merge all fragments in the same skeleton cluster -> skeleton_cluster,
        and merge branch neighbor (only one node) to skeleton node,
        or merge branch neighbors together to form an independent node -> branch_cluster
        :param frag_cluster: a list of Node class
        :return: {'sk_cluster': a single Node obj, 'br_cluster': a list of Node obj}
        """
        # get all possible fragment cluster
        skeleton_premerge_obj = PreMerge(cluster_type='skeleton', smiles=self.smiles)  # all SMILES of skeleton node
        branch_cluster = []  # a list of Node obj
        for node in frag_cluster:
            branch_path = node.branch_path  # [[2, 1]]
            _result = deal_with_branch_path(current_node=node, branch_path=branch_path,
                                            smiles=self.smiles, id2node=self.id2node)
            branch_cluster += _result['branch_node']
            skeleton_premerge_obj.add_node(_result['current_node'])

        skeleton_cluster = skeleton_premerge_obj.merge()  # a single Node obj
        return {'sk_cluster': skeleton_cluster, 'br_cluster': branch_cluster}

    def split_by_long_branch(self, tmp_clusters):
        # split by node which has long branch, test data LINE 95
        tmp_clusters_new = []
        for _tmc in tmp_clusters:
            num_branch_path = []
            # _tmc = [int(i) for i in _tmc]
            for _each_id in _tmc:
                _each_id = int(_each_id)
                neighbor_branch = self.id2node[_each_id].branch_path
                if len(neighbor_branch) == 0:
                    num_branch_path.append(0)
                else:
                    # print(neighbor_branch)
                    num_branch_path.append(np.max([len(i) for i in neighbor_branch]))
            long_branch_bool = np.array(num_branch_path) >= 2
            long_branch_inx = np.where(long_branch_bool)[0]

            _frag_new = []
            if (np.sum(long_branch_bool) >= 2) and (len(_tmc) > 3):
                for i in range(len(_tmc)):
                    if i in long_branch_inx:
                        _frag_new.append(_tmc[i])
                        tmp_clusters_new.append(_frag_new)
                        _frag_new = []
                    else:
                        _frag_new.append(_tmc[i])
                    if (i == len(_tmc) - 1) and _frag_new:  # test data LINE 391
                        tmp_clusters_new.append(_frag_new)
            else:
                tmp_clusters_new.append(_tmc)
        return tmp_clusters_new

    def split_mol_path(self, min_len=5):
        clusters_by_ring = self.split_by_ring()
        # print('_clusters:', _clusters)
        short_clusters = []
        for _c in clusters_by_ring:
            _tmp_clusters = split_long_cluster(_c, min_len=min_len)
            short_clusters += self.split_by_long_branch(_tmp_clusters)
        return short_clusters

    def do(self, test=False, min_len=6):
        short_clusters = self.split_mol_path(min_len=min_len)
        if test:
            print('Short fragment: ', short_clusters)  # [['1', '12', '3'], ['4', '5', '6']]
        merged_cluster_nodes = []
        for sf_inx in range(len(short_clusters)):
            single_cluster = [int(i) for i in short_clusters[sf_inx]]
            _current_inx_exist = False
            frag_cluster = [self.id2node[i] for i in single_cluster]  # a list of Node this cluster
            merged_frag = self.merge_frags(frag_cluster)  # {'sk_cluster': Node, 'br_cluster': [Node]}
            merged_cluster_nodes.append(merged_frag['sk_cluster'])
            if len(merged_frag['br_cluster']) >= 1:
                merged_cluster_nodes += merged_frag['br_cluster']

            if test:
                print('\n')
                print('>>> each cluster >>>')
                print('    sk node: ', merged_frag['sk_cluster'].show_info())
                print('    br node: ', [(i.id, i.smiles) for i in merged_frag['br_cluster'] if i])
        # if test:
        #         #     for node in merged_cluster_nodes:
        #         #         node.show_info()
        merged_cluster_info = []
        # merge short cluster which the number of atoms <= 2
        merged_cluster_nodes_bak = merged_cluster_nodes.copy()
        for i in range(len(merged_cluster_nodes_bak)):
            curr_node = merged_cluster_nodes_bak[i]
            if curr_node.num_atom <= 2:
                new_permerge_obj = PreMerge(cluster_type='skeleton', smiles=self.smiles)
                new_permerge_obj.add_node(curr_node)
                merged_cluster_nodes[i] = None
                # merge_i = i - 1
                if i == 0 or (merged_cluster_nodes_bak[i-1].type == 'branch'):
                    # i == 0 means the first cluster, or
                    # previous cluster is branch, such as cid: 81422426, got following unexpected result
                    # {'mol_tree_id': '6-3-4', 'smiles': 'CC.CNC', 'mol_inx': [3, 9, 5, 6, 7], 'type': 'skeleton'}
                    merge_i = i + 1  # following cluster
                else:
                    merge_i = i - 1  # previous cluster
                new_permerge_obj.add_node(merged_cluster_nodes[merge_i])
                merged_cluster_nodes[merge_i] = new_permerge_obj.merge()

        for node in merged_cluster_nodes:
            if node is not None:
                merged_cluster_info.append({'mol_tree_id': str(node.id),
                                            'smiles': node.smiles,
                                            'mol_inx': node.mol_inx,
                                            'type': node.type})

        return merged_cluster_info


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def basic_test():
    SMILES = 'C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F'
    mol = Chem.MolFromSmiles(SMILES)
    id2smiles = {1: 'C#C', 2: 'CC', 3: 'CN', 4: 'CN', 5: 'CC', 6: 'C#C', 7: 'CN',
                 8: 'C=O', 9: 'CC', 10: 'CO', 11: 'CO', 12: 'CF', 13: 'CF',
                 14: 'C1=CCCC=C1', 15: 'C1=CC=CC=C1', 16: 'N', 17: 'C', 18: 'C'}
    n2n = {1: [2], 2: [1, 3], 3: [2, 16], 4: [5, 16], 5: [4, 6], 6: [5],
           7: [16, 17], 8: [17], 9: [14, 17], 10: [11, 14], 11: [10, 18], 12: [18],
           13: [18], 14: [9, 10, 15], 15: [14], 16: [3, 4, 7], 17: [7, 8, 9], 18: [11, 12, 13]}
    g = mol2network(n2n, file_path='.', name='3434', draw_network=True)
    # mol_path = get_mol_path(n2n, g)

    print('>>> SMILES: ', SMILES)
    print('    n2n: ', n2n)
    draw_graph(g, file_dir='.', file_name='test')


def writh_log_message(mes, path):
    with open(path, 'a') as f:
        f.write(mes + '\n')


if __name__ == '__main__':
    testLines = 10
    min_len = 5
    parser = argparse.ArgumentParser(
        description='molecule fragment by tree decomposition')
    parser.add_argument('input_fn',
                        help='training set file path')
    parser.add_argument('result_fn',
                        help='result file path')
    parser.add_argument('--log_fn',
                        help='log file name')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run the entire script on only the first {} lines and plot.'.format(testLines))
    args = parser.parse_args()
    test = args.test
    log_file = args.log_fn
    input_file = args.input_fn
    result_file = args.result_fn
    if test:
        if not os.path.exists('./big-data/figure'):
            os.makedirs('./big-data/figure')
    with open(input_file, 'r') as input_f:
        counter = 0
        for row in input_f:
            if test and counter > testLines:
                break
            row = row.strip().split('\t')
            if row[0] == 'cid':
                current_row = row
            else:
                current_row = [json.loads(i) for i in row]
            try:
                cid = current_row[0]
            except KeyError:
                cid = ''
            if row[0] != 'cid':  # the first line
                SMILES = current_row[1]
                id2smiles = {int(i): j for i, j in current_row[2].items()}  # frag_id2smiles
                n2n = {int(i): j for i, j in current_row[3].items()}  # frag_id2neighbors
                id2mol_inx = {int(i): j for i, j in current_row[4].items()}  # frag_id2mol_inx
                if counter % 500000 == 0:
                    print('>>> current line: ', counter)
                    print('>>>CID: {}, SMILES: (line {})'.format(cid, counter), SMILES)
                network = Mol2Network(n2n, id2smiles)
                if test:
                    print('    id2smiles: ', id2smiles)
                    print('    n2n: ', n2n)
                    draw_graph(network.g, file_dir=os.path.join('big-data', 'figure'),
                               file_name='mol_tree_cid:{}_line:{}'.format(cid, counter))
                    mol = Chem.MolFromSmiles(SMILES)
                    Draw.MolToFile(mol, os.path.join('big-data', 'figure',
                                                     'mol_structure_{}.png'.format(counter)))
                    mol_with_inx = mol_with_atom_index(mol)
                    Draw.MolToFile(mol_with_inx,
                                   os.path.join('big-data', 'figure',
                                                'mol_with_inx_{}.png'.format(counter)))
                try:
                    mol_path = network.get_mol_path()
                    if counter % 500000 == 0:
                        print('>>> current line: ', counter)
                        print('    Mol path: ', mol_path)
                    id2node = network.get_id2node(mol_path, id2smiles, id2mol_inx)
                    if test:
                        print('    New SMILES: ', {i: id2node[i].smiles for i in list(id2node.keys())})
                    refragmenter = Refragment(mol_path, id2node, SMILES, log_file, n2n)
                    merged_clusters_info = refragmenter.do(test, min_len=min_len)
                except Exception as e:
                    merged_clusters_info = []
                    mol_path = []
                    with open(log_file, 'a') as log_f:
                        log_f.write('refragment error, cid: {}'.format(cid) + '\n')
                if merged_clusters_info:
                    if counter % 500000 == 0 or test:
                        print('>>> current line: ', counter)
                        print('    Valid indices of clusters: ')
                        print(merged_clusters_info)
                    mol_inx = [len(i['mol_inx']) == 0 for i in merged_clusters_info]
                    if np.any(mol_inx):
                        writh_log_message('cid: {}, SMILES: {}, some clusters short index '
                                          'in molecule structure'.format(cid, SMILES), log_file)
                    # if merged_clusters_info:
                    with open(result_file, 'a') as f:
                        mol_path_str = ','.join([str(i) for i in mol_path])
                        inx_keys = ','.join([i['mol_tree_id'] for i in merged_clusters_info])
                        frag_smiles = ','.join([i['smiles'] for i in merged_clusters_info])
                        f.write('\t'.join([cid, mol_path_str, inx_keys, frag_smiles]) + '\n')
            counter += 1
