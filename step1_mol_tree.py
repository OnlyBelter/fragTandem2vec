"""
The first step: generate molecular tree based on https://github.com/wengong-jin/icml18-jtnn
Then we can get the fragments of each molecule and the relation between each two fragments
# usage:
$ python step1_mol_tree.py demo_data/demo_dataset.txt demo_data/step1_result.txt --log_fn demo_data/step1_log.log
"""
from __future__ import print_function
import rdkit
import rdkit.Chem as Chem
from chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, set_atommap, enum_assemble
from pub_func import if_only_main_element, write_list_by_json
import json
import argparse


class MolTreeNode(object):

    def __init__(self, smiles, clique=[]):
        self.smiles = smiles
        self.mol = get_mol(self.smiles)

        self.clique = [x for x in clique]  # copy
        self.neighbors = []

    def add_neighbor(self, nei_node):
        self.neighbors.append(nei_node)

    def recover(self, original_mol):
        clique = []
        clique.extend(self.clique)
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            for cidx in nei_node.clique:
                # allow singleton node override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        self.label = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label

    def assemble(self):
        neighbors = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands, aroma = enum_assemble(self, neighbors)
        new_cands = [cand for i, cand in enumerate(cands) if aroma[i] >= 0]
        if len(new_cands) > 0:
            cands = new_cands

        if len(cands) > 0:
            self.cands, _ = zip(*cands)
            self.cands = list(self.cands)
        else:
            self.cands = []


class MolTree(object):

    def __init__(self, smiles, common_atom_split_ring=3):
        self.smiles = smiles
        self.mol = get_mol(smiles)

        # Stereo Generation (currently disabled)
        # mol = Chem.MolFromSmiles(smiles)
        # self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        # self.smiles2D = Chem.MolToSmiles(mol)
        # self.stereo_cands = decode_stereo(self.smiles2D)

        cliques, edges = tree_decomp(self.mol, common_atom_split_ring)
        self.nodes = []
        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
            self.nodes.append(node)
            if min(c) == 0: root = i

        for x, y in edges:
            self.nodes[x].add_neighbor(self.nodes[y])
            self.nodes[y].add_neighbor(self.nodes[x])

        if root > 0:
            self.nodes[0], self.nodes[root] = self.nodes[root], self.nodes[0]

        for i, node in enumerate(self.nodes):
            node.nid = i + 1
            if len(node.neighbors) > 1:  # Leaf node mol is not marked
                set_atommap(node.mol, node.nid)
            node.is_leaf = (len(node.neighbors) == 1)

    def size(self):
        return len(self.nodes)

    def recover(self):
        for node in self.nodes:
            node.recover(self.mol)

    def assemble(self):
        for node in self.nodes:
            node.assemble()


def dfs(node, fa_idx):
    max_depth = 0
    for child in node.neighbors:
        if child.idx == fa_idx:
            continue
        max_depth = max(max_depth, dfs(child, node.idx))
    return max_depth + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='molecule fragment by tree decomposition')
    parser.add_argument('training_set_fn',
                        help='training set file path')
    parser.add_argument('result_fn',
                        help='result file path')
    parser.add_argument('--log_fn',
                        help='log file name')

    args = parser.parse_args()

    # root_dir = r'/home/belter/github/my-research/jtnn-py3'
    raw_data = args.training_set_fn
    # result_dir = args.result_fn
    result_file = args.result_fn
    log_file = args.log_fn
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    cset2count = {}  # count the number of fragment
    columns = ['cid', 'smiles', 'frag_id2smiles', 'frag_id2neighbors', 'frag_id2mol_inx']
    # result = pd.DataFrame()
    with open(result_file, 'w') as f:
        f.write('\t'.join(columns) + '\n')
    with open(raw_data) as f:
        counter = 1
        for line in f:
            if counter % 5000 == 0:
                print('>>> current line: ', counter)
            node2neighbors = {}
            mol_blocks = {}  # id2smiles
            id2mol_inx = {}
            _line = line.strip().split('\t')
            # print(_line)
            if len(_line) == 2:
                cid, smiles = _line
            elif len(_line) == 1:
                cid = 'id' + str(counter)
                smiles = _line[0]
            else:
                raise Exception('Each line should separate by tab and 1 or 2 columns')
            # cid, smiles = line.strip().split('\t')
            if cid.lower() != 'cid':
                only_main_ele = if_only_main_element(smiles)
                if only_main_ele:
                    try:
                        mol = MolTree(smiles, common_atom_split_ring=3)
                        for i in mol.nodes:
                            if i not in mol_blocks:
                                mol_blocks[i.nid] = ''
                                id2mol_inx[i.nid] = []
                            mol_blocks[i.nid] = i.smiles
                            id2mol_inx[i.nid] = i.clique
                        for node in mol.nodes:
                            if node not in node2neighbors:
                                node2neighbors[node.nid] = []
                            node2neighbors[node.nid] += [i.nid for i in node.neighbors]
                        with open(result_file, 'a') as result_f:
                            write_str = write_list_by_json([cid, smiles, mol_blocks, node2neighbors, id2mol_inx])
                            result_f.write(write_str)
                    except Exception as e:
                        with open(log_file, 'a') as log_f:
                            log_f.write('mol_tree error, cid: {}'.format(cid) + '\n')
                else:
                    with open(log_file, 'a') as log_f:
                        log_f.write('This molecule contains rare elements: {}, {}'.format(cid, smiles) + '\n')
                counter += 1
