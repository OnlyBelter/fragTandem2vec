"""
step2: generate fragment sentence
# usage:
$ python refragment.py ./demo_data/step1_result.txt ./demo_data --log_fn ./demo_data/step2_log.log
# plot molecular structure, molecular tree and molecular with index of the first 10 lines under test model
$ python refragment.py ./demo_data/step1_result.txt ./demo_data --log_fn ./demo_data/step2_log.log --test
"""
import networkx as nx
import matplotlib.pyplot as plt
import os
from rdkit import Chem
from rdkit.Chem import Draw
import json
from pub_func import Mol2Network
from helper_func import replace_smiles_by_frag_id, count_fragment
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
                        help='file path of molecular fragments information from the first step, '
                             'which contains cid/smiles/frag_id2smiles/frag_id2neighbors/frag_id2mol_inx')
    parser.add_argument('result_dir',
                        help='result directory')
    parser.add_argument('--log_fn',
                        help='log file name')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Run the entire script on only the first {} lines and plot.'.format(testLines))
    args = parser.parse_args()
    test = args.test
    log_file = args.log_fn
    input_file = args.input_fn
    result_dir = args.result_dir
    result_file_cid2frag = os.path.join(result_dir, 'step2_cid2smiles_sentence.csv')
    result_file_frag2num = os.path.join(result_dir, 'step2_frag2num.csv')
    result_file_cid2frag_id = os.path.join(result_dir, 'step2_cid2frag_id_sentence.csv')
    result_file_frag_id_sentence = os.path.join(result_dir, 'step2_frag_id_sentence.csv')
    if test:
        if not os.path.exists('./big-data/figure'):
            os.makedirs('./big-data/figure')
    print('>>> Start to refragment...')
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
            if row[0] != 'cid':  # remove the first line
                SMILES = current_row[1]
                id2smiles = {int(i): j for i, j in current_row[2].items()}  # frag_id2smiles
                n2n = {int(i): j for i, j in current_row[3].items()}  # frag_id2neighbors
                id2mol_inx = {int(i): j for i, j in current_row[4].items()}  # frag_id2mol_inx
                if counter % 500000 == 0 or test:
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

                mol_sentence = []
                try:
                    mol_paths = network.get_mol_path()  # all mol paths
                    if counter % 500000 == 0:
                        print('>>> current line: ', counter)
                        print('    The longest mol path of this molecule: ', mol_paths[0])
                    for mol_path in mol_paths:
                        mol_sentence += [id2smiles[frag_id] for frag_id in mol_path]
                except Exception as e:
                    with open(log_file, 'a') as log_f:
                        log_f.write('refragment error, cid: {}'.format(cid) + '\n')

                if mol_sentence:
                    if counter % 500000 == 0 or test:
                        # print('>>> current line: ', counter)
                        print('    Mol sentence: ')
                        print(mol_sentence)
                        print('    mol paths: ', mol_paths)
                    with open(result_file_cid2frag, 'a') as f:
                        # mol_path_str = ','.join([str(i) for i in mol_paths])
                        frag_smiles = ','.join(mol_sentence)
                        f.write('\t'.join([cid, frag_smiles]) + '\n')
            counter += 1

    # count fragment frequency
    print('>>> Start to count fragments...')
    cid2frag_file_path = result_file_cid2frag
    frag2num = count_fragment(cid2frag_file_path)
    frag2num.to_csv(result_file_frag2num, index_label='frag_id')

    # replace fragment smiles with fragment id
    print('>>> Start to replace fragment SMILES by fragment id...')
    replace_smiles_by_frag_id(result_file_frag2num, cid2frag_file_path,
                              result_fp=result_file_cid2frag_id, result_fp2=result_file_frag_id_sentence)
