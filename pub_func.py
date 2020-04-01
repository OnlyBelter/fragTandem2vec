import networkx as nx
from rdkit import Chem
import numpy as np
from itertools import product
from operator import itemgetter
import json
import datetime


MAIN_ELEMENT = ['C', 'O', 'N', 'H', 'P', 'S', 'Cl', 'F', 'Br', 'I']


def get_mol_obj(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
    except TypeError:
        m = Chem.MolFromSmarts(smiles)
    return m


def get_num_atom_by_smiles(smiles):
    m = get_mol_obj(smiles)
    num_atom = m.GetNumAtoms()
    return num_atom


def get_smiles(mol):
    """
    mol obj -> SMILES
    :param mol:
    :return:
    """
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def get_mol(smiles):
    """
    SMILES -> mol obj
    :param smiles:
    :return:
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def get_smiles_by_inx(smiles, inx_cluster):
    """

    :param smiles: the SMILES of original mol
    :param inx_cluster: a set of atom index in molecule, at least contains two elements
    :return:
    """
    # atommap = {}
    #     # mol = Chem.MolFromSmiles(smiles)
    #     # bonds = []
    #     # for i, j in combinations(inx_cluster, 2):
    #     #     b = mol.GetBondBetweenAtoms(i, j)
    #     #     if b:
    #     #         bonds.append(b.GetIdx())
    #     # new_mol = Chem.PathToSubmol(mol, bonds, atomMap=atommap)
    #     # return Chem.MolToSmiles(new_mol)
    mol = get_mol(smiles)
    smiles = Chem.MolFragmentToSmiles(mol, inx_cluster, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    # new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return get_smiles(new_mol)


def get_mol_inx_from_node_list(node_list):
    """

    :param node_list: a list of Node obj
    :return:
    """
    assert type(node_list) == list
    mol_inx = []
    for node in node_list:
        for inx in node.mol_inx:
            if inx not in mol_inx:
                mol_inx.append(inx)
    return mol_inx


class Node:
    def __init__(self, tree_id, smiles, ring,
                 num_atom, mol_inx, node_type, branch_path):
        self.id = tree_id
        self.smiles = smiles  # SMILES of this node (fragment/cluster)
        self.ring = ring  # True/False
        self.num_atom = num_atom  # int
        self.mol_inx = mol_inx  # a list
        self.neighbors = []  # all neighbors of this node
        self.type = node_type  # the type of this node, skeleton/branch
        self.branch_path = branch_path  # a list of node id

    def show_info(self):
        print('>>>>')
        print('id: ', self.id)
        print('smiles: ', self.smiles)
        print('num atom: ', self.num_atom)
        print('type: ', self.type)
        print('neighbors: ', self.neighbors)
        print('mol_inx: ', self.mol_inx)
        print()

    def add_neighbor(self, neighbor):
        # neighbor is a node class
        self.neighbors.append(neighbor)


class PreMerge:
    def __init__(self, cluster_type, smiles, sk_neighbor=None):
        self.smiles = smiles  # SMILES of this whole molecule
        self.type = cluster_type  # skeleton or branch
        self.nodes = []  # a list of Node
        self.sk_neighbor_id = sk_neighbor  # a node, which is the neighbor of this branch cluster

    def add_node(self, node):
        self.nodes.append(node)

    def merge(self):
        if len(self.nodes) == 0:
            return None
        if len(self.nodes) == 1:
            return self.nodes[0]
        else:
            mol_inx = get_mol_inx_from_node_list(self.nodes)
            tree_id = '-'.join([str(i.id) for i in self.nodes])
            smiles = get_smiles_by_inx(self.smiles, mol_inx)
            ring = np.any([i.ring for i in self.nodes])
            num_atom = len(mol_inx)
            node_type = self.type
            branch_path = []
            merged_node = Node(tree_id=tree_id, smiles=smiles, ring=ring, num_atom=num_atom,
                               mol_inx=mol_inx, node_type=node_type, branch_path=branch_path)
            return merged_node


def check_ring(smiles):
    """
    check if a SMILES is ring
    :param smiles:
    :return:
    """
    # https://github.com/rdkit/rdkit/issues/1984
    m = Chem.MolFromSmarts(smiles)
    m.UpdatePropertyCache()
    Chem.GetSymmSSSR(m)
    ring_info = m.GetRingInfo()
    if ring_info.NumRings() >= 1:
        return True
    else:
        return False


def check_in_ring(mol, atom_inx):
    """
    check if atom_inx is in a ring
    :param mol: Chem.MolFromSmiles(smiles)
    :param atom_inx:
    :return:
    """
    if mol.GetAtomWithIdx(atom_inx).IsInRing():
        return True
    else:
        return False


def check_valid_inx_new(cluster_smiles, frag2inx, mol, taken_inx=(), current_line_inx=1):
    """
    filter the index of the next neighbor by previous index,
    a neighbor pair should have at least a common index (atom)
    :param cluster_smiles: ['CN', 'C=O', 'CC'], SMILES of this cluster in order
    :param frag2inx: {'CN': ((1, 0), (7, 8)), 'C=O': ((1, 2),),
                      'CC': ((1, 3), (3, 4), (3, 25), (4, 5), (5, 6), (6, 7), (6, 24), (24, 25))}
    :param mol: Mol class in rdkit, self.mol = Chem.MolFromSmiles(smiles)
    :param taken_inx: used indies by other fragments
    :param current_line_inx: index of current line
    :return: valid order2inx, [{0: inx, 1: inx, ...}, {0: inx, 1: inx, ...}]
    """
    valid_order2inx = []  # may more than one valid order2inx
    unique_inx = []  # only save the sorted unique index of this cluster
    # first_indices = frag2inx[cluster_smiles[0]]
    inx_list = []
    len_cluster = len(cluster_smiles)

    for smiles in cluster_smiles:
        _inx = frag2inx[smiles]
        if (not check_ring(smiles)) and taken_inx:  # test data LINE 14
            if current_line_inx == 0:  # the first node in this cluster, test data LINE 13
                _inx = set([i for i in _inx if len(set(i).intersection(set(taken_inx))) == 1])
            else:
                _inx = set([i for i in _inx if len(set(i).intersection(set(taken_inx))) <= 1])
        # if len(_inx) > 1:
        #     _inx = frag2inx[smiles][0]
        inx_list.append(_inx)
    total_path = list(product(*inx_list))  # all combinations of each fragment

    # one smiles may have two indices, such as {'CN': ((6, 5), (12, 13))}
    if len_cluster == 1:
        for i in range(len(total_path)):
            valid_order2inx.append({0: total_path[i][0]})  # the key is always 0
        return valid_order2inx

    # only one possible combination in this cluster
    if len(total_path) == 1:
        valid_order2inx.append({i: total_path[0][i] for i in range(len(total_path[0]))})
        return valid_order2inx

    for path in total_path:
        if len(set(path)) != len_cluster:
            continue  # returns the control to the beginning of the while loop
        first_inx = path[0]
        last_inx = path[-1]
        # avoid both two continue atoms are in the same ring instead of in skeleton path
        # {0: (2, 14), 1: (14, 15), 2: (14, 16), 3: (17, 16)} for ['CC', 'C', 'CN', 'CN'] is not valid
        # in molecule "CCC(NC(=O)c1scnc1C1CC1)C(=O)N1CCOCC1"
        if len(first_inx) == 2:
            if check_in_ring(mol, first_inx[0]) and check_in_ring(mol, first_inx[-1]):
                continue
        if len(last_inx) == 2:
            if check_in_ring(mol, last_inx[0]) and check_in_ring(mol, last_inx[-1]):
                continue

        current_valid_path = []

        if len(path) == 1:
            current_valid_path.append(path[0])
        else:
            _valid_marker = True
            for i in range(len(path) - 1):
                current_inx = path[i]
                next_inx = path[i + 1]
                num_intersection = sum([k in current_inx for k in next_inx])
                if check_ring(cluster_smiles[i]) or check_ring(cluster_smiles[i + 1]):
                    if not (1 <= num_intersection <= 2):
                        _valid_marker = False
                else:  # test data LINE 14
                    if not (num_intersection == 1):
                        _valid_marker = False
            if _valid_marker:
                current_valid_path = list(path)

        if len(current_valid_path) == len_cluster:
            _order2inx = {i: current_valid_path[i] for i in range(len_cluster)}
            sorted_inx = sorted(_order2inx.values())
            # remove duplicated items, such as [{0: (0, 1), 1: (1, 2)}, {0: (1, 2), 1: (0, 1)}]
            if sorted_inx not in unique_inx:
                unique_inx.append(sorted_inx)

            valid_order2inx.append(_order2inx)
    return valid_order2inx


def get_frag_inx(smiles, frags):
    """
    get index of fragments in molecule
    :param smiles: the SMILES of molecule
    :param frags: a list of SMILES
    :return: the index of each atom in this fragment, {frag1: (2, 3), frag2: (7, 8, 9)}
    """
    mol = Chem.MolFromSmiles(smiles)
    all_indices = {}
    for frag in frags:
        if frag not in all_indices:
            # find SMILES first
            try:
                inx = mol.GetSubstructMatches(Chem.MolFromSmiles(frag))
            except TypeError:
                inx = ''
            if not inx:
                try:
                    inx = mol.GetSubstructMatches(Chem.MolFromSmarts(frag))
                except TypeError:
                    inx = ''
            all_indices[frag] = inx
    return all_indices


class Mol2Network:
    def __init__(self, smiles, n2n, id2smiles, id2mol_inx):
        """
        represent a molecule by graph
        :param smiles: the SMILES of this molecule, eg: C#CCN(CC#C)C(=O)c1cc2ccccc2cc1OC(F)F
        :param n2n: node to neighbors (node id and it's neighbors)
        - {1: [2], 2: [1, 3], 3: [2, 16], 4: [5, 16], 5: [4, 6], 6: [5],
           7: [16, 17], 8: [17], 9: [14, 17], 10: [11, 14], 11: [10, 18], 12: [18],
           13: [18], 14: [9, 10, 15], 15: [14], 16: [3, 4, 7], 17: [7, 8, 9], 18: [11, 12, 13]}
        :param id2smiles: fragment id 2 fragment smiles, eg:
        - {1: 'C#C', 2: 'CC', 3: 'CN', 4: 'CN', 5: 'CC', 6: 'C#C', 7: 'CN',
           8: 'C=O', 9: 'CC', 10: 'CO', 11: 'CO', 12: 'CF', 13: 'CF',
           14: 'C1=CCCC=C1', 15: 'C1=CC=CC=C1', 16: 'N', 17: 'C', 18: 'C'}
        :param id2mol_inx: fragment id to atom index in this molecule, eg:
        - {"1": [0, 1], "2": [1, 2], "3": [2, 3], "4": [3, 4], "5": [4, 5], "6": [5, 6], "7": [3, 7],
           "8": [7, 8], "9": [7, 9], "10": [18, 19], "11": [19, 20], "12": [20, 21], "13": [20, 22],
           "14": [9, 18, 17, 16, 11, 10], "15": [12, 13, 14, 15, 16, 11], "16": [3], "17": [7], "18": [20]}
        """
        self.smiles = smiles
        self.n2n = n2n
        self.id2smiles = id2smiles
        self.id2mol_inx = id2mol_inx
        self.g = self._get_graph()
        self.end_points = self._get_end_points()

    def _get_graph(self):
        # network of molecule generated by networkx
        g = nx.Graph()
        for i in self.n2n.keys():
            edges = [(i, j) for j in self.n2n[i]]
            g.add_edges_from(edges)

        id2smile_attr = {k: {'smiles': v} for k, v in self.id2smiles.items()}
        id2mol_inx_attr = {int(k): {'mol_inx': v} for k, v in self.id2mol_inx.items()}
        nx.set_node_attributes(g, id2smile_attr)
        nx.set_node_attributes(g, id2mol_inx_attr)
        return g

    def _get_end_points(self):
        end_points = [i for i in self.n2n if len(self.n2n[i]) == 1]
        return end_points

    def _get_end_pairs(self):
        # get end point (only one neighbor) pairs (all combination)
        end_points = self.end_points
        num_ = len(end_points)
        end_pairs = []
        if num_ > 2:
            for i in range(num_):
                for j in range(num_):
                    if i < j:
                        end_pairs.append((end_points[i], end_points[j]))
        else:
            end_pairs.append(tuple(end_points))
        return end_pairs

    def count_neighbors(self, mol_path):
        num_neighbors = 0
        for i in mol_path:
            _n_n = len(self.n2n[i])
            num_neighbors += _n_n
        return num_neighbors

    def get_mol_path(self):
        """
        get all molecular paths from end to end (end pairs of atom in molecule)
        :return:
        """
        end_pairs = self._get_end_pairs()  # get end point pairs
        paths_with_attr = []  # with attribute
        all_paths = []
        num_node_longest_path = 0
        num_max_path_neighbors = 0
        # num_max_atom_in_path = 0
        if len(self.n2n) >= 2:
            for pairs in end_pairs:
                # print(pairs)
                shortest_path = nx.shortest_simple_paths(self.g, pairs[0], pairs[1])
                shortest_path = list(shortest_path)[0]
                num_neig = self.count_neighbors(shortest_path)
                num_atoms = np.sum([get_num_atom_by_smiles(self.id2smiles[i]) for i in shortest_path])
                paths_with_attr.append({'path': shortest_path, 'len_path': len(shortest_path),
                                  'num_neig': num_neig, 'num_atoms': num_atoms})
                # print(shortest_path)
                if len(shortest_path) > num_node_longest_path:
                    num_node_longest_path = len(shortest_path)
                if num_neig > num_max_path_neighbors:
                    num_max_path_neighbors = num_neig
                # if num_atoms > num_max_atom_in_path:
                #     num_max_atom_in_path = num_atoms
            paths_with_attr = sorted(paths_with_attr, key=itemgetter('num_atoms'), reverse=True)
            paths_with_attr = sorted(paths_with_attr, key=itemgetter('num_neig'), reverse=True)  # test data LINE 401
            for path in paths_with_attr:
                all_paths.append(path['path'])
        if len(self.n2n) == 1:  # only one node in all graph, test data LINE 546
            all_paths = list(self.n2n.keys())
        return all_paths

    def get_id2node(self, mol_path, id2smile, id2mol_inx):
        """
        fragment id maps to Node class
        :param mol_path: mol-tree id in the longest path (skeleton), a list
        :param id2smile: id to SMILES
        :param id2mol_inx: id to index in molecule
        :return: {id: Node class, ...}
        """
        id2node = {}
        for i, smiles in id2smile.items():
            ring = check_ring(smiles)
            num_atom = get_num_atom_by_smiles(smiles)
            mol_inx = id2mol_inx[i]
            node_type = 'branch'
            branch_path = []
            if i in mol_path:
                node_type = 'skeleton'
                branch_path = self.get_skeleton_node_branch_path(mol_path, i)
            node = Node(tree_id=i, smiles=smiles, ring=ring, num_atom=num_atom,
                        mol_inx=mol_inx, node_type=node_type, branch_path=branch_path)
            id2node[i] = node
        for i, node in id2node.items():
            neis_tree_id = self.n2n[i]
            for nei in neis_tree_id:
                node.add_neighbor(id2node[nei])

        return id2node

    def get_skeleton_node_branch_path(self, mol_path, frag_id):
        """

        :param mol_path: all fragments on the skeleton
        :param frag_id: fragment id in mol tree
        :return: branch_path, a list of node id
        """
        direct_neighbors = self.n2n[frag_id]
        branch_path = []
        if len(direct_neighbors) >= 3:  # usually have 1 branch
            branch_neighbors = [i for i in direct_neighbors if i not in mol_path]
            # all_neighbors['skeleton_neighbors'] += [i for i in direct_neighbors if i in mol_path]
            for j in branch_neighbors:
                if j not in self.end_points:
                    for ep in self.end_points:
                        branch_path_tmp = nx.shortest_simple_paths(self.g, frag_id, ep)  # test data LINE 14
                        branch_path_tmp = list(branch_path_tmp)[0]
                        # remove frag_id, which is included in skeleton
                        branch_path_tmp = [node_id for node_id in branch_path_tmp if node_id != frag_id]
                        branch_path_tmp_bak = branch_path_tmp.copy()
                        if j in branch_path_tmp_bak:
                            # add other neighbors in this branch path
                            for branch_node in branch_path_tmp_bak:
                                branch_node_neighs = self.n2n[branch_node]
                                branch_node_neighs = [node_id for node_id in branch_node_neighs
                                                      if node_id not in mol_path]
                                for branch_node_neig in branch_node_neighs:
                                    if branch_node_neig not in branch_path_tmp_bak:
                                        branch_path_tmp.append(branch_node_neig)
                            branch_path.append(branch_path_tmp)
                            break
                else:
                    branch_path.append([j])  # only one node in this branch path

        return branch_path


def if_only_main_element(smiles):
    only_main_element = True
    fr_mol = get_mol(smiles)
    if fr_mol:
        for atom in fr_mol.GetAtoms():
            if atom.GetSymbol() not in MAIN_ELEMENT:
                only_main_element = False
                break
    return only_main_element


def write_list_by_json(a_list, sep='\t'):
    return sep.join([json.dumps(i) for i in a_list]) + '\n'


def get_format_time():
    t = datetime.datetime.now()
    return t.strftime("%c")
