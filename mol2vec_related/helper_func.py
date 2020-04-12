"""
helper function for mol2vec_related
"""
import os
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import datetime
from joblib import Parallel, delayed
from gensim.models import word2vec
from mol2vec.features import generate_corpus, train_word2vec_model, mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec


def get_cid2smiles(cid2smiles, cid_list, result_file):
    """
    get cid2smiles.txt in training set
    :param cid2smiles: file path, cid, smiles
    :param cid_list： file path of cid list in train_set
    :param result_file: file path of result
    :return:
    """
    cid2smiles_dict = {}
    with open(cid2smiles, 'r') as f_handle:
        for i in f_handle:
            i = i.strip()
            cid, smiles = i.split('\t')
            cid2smiles_dict[cid] = smiles
    with open(cid_list, 'r') as f_handle2:
        for i in f_handle2:
            i = i.strip()
            cid = i.split('\t')[0]
            if cid in cid2smiles_dict:
                with open(result_file, 'a') as r_handle:
                    r_handle.write(cid + '\t' + cid2smiles_dict[cid] + '\n')
            else:
                print('>>> this compound {} does not exist in our cid2smiles.txt list...'.format(cid))


def generate_corpus_from_smiles(in_file, out_file, r, sentence_type='alt', n_jobs=1):
    """
    modified from generate_corpus
    https://mol2vec.readthedocs.io/en/latest/#mol2vec.features.generate_corpus
    :param in_file: cid, smiles
    :param out_file:
    :param r: int, Radius of morgan fingerprint
    :param sentence_type:
    :param n_jobs:
    :return:
    """
    all_smiles = []
    with open(in_file, 'r') as f_handle:
        for each_line in f_handle:
            if ',' in each_line:
                cid, smiles = each_line.strip().split(',')
            else:
                cid, smiles = each_line.strip().split('\t')
            if smiles != 'smiles':
                all_smiles.append(smiles)

    if sentence_type == 'alt':  # This can run parallelized
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(smiles, r) for smiles in all_smiles)
        for i, line in enumerate(result):
            with open(out_file, 'a') as f_handle:
                f_handle.write(str(line) + '\n')
        print('% molecules successfully processed.')


def _parallel_job(smiles, r):
    """Helper function for joblib jobs
    """
    if smiles is not None:
        # smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, r)
        return " ".join(sentence)


def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius

    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def _read_corpus(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield line.split()


def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK'):
    """Handling of uncommon "words" (i.e. identifiers). It finds all least common identifiers (defined by threshold) and
    replaces them by 'uncommon' string.

    Parameters
    ----------
    corpus : str
        Input corpus file
    out_corpus : str
        Outfile corpus file
    threshold : int
        Number of identifier occurrences to consider it uncommon
    uncommon : str
        String to use to replace uncommon words/identifiers

    Returns
    -------
    """
    # Find least common identifiers in corpus
    f = open(corpus)
    unique = {}
    for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
        for identifier in x:
            if identifier not in unique:
                unique[identifier] = 1
            else:
                unique[identifier] += 1
    n_lines = i + 1
    least_common = set([x for x in unique if unique[x] <= threshold])
    f.close()

    f = open(corpus)
    fw = open(out_corpus, mode='w')
    for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
        intersection = set(line) & least_common
        if len(intersection) > 0:
            new_line = []
            for item in line:
                if item in least_common:
                    new_line.append(uncommon)
                else:
                    new_line.append(item)
            fw.write(" ".join(new_line) + '\n')
        else:
            fw.write(" ".join(line) + '\n')
    f.close()
    fw.close()


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


def load_trained_model(model_fp):
    """
    load well-trained model by following function
    train_word2vec_model('./mols_demo_corpus.txt', outfile_name='mols_demo_model.pkl',
                          vector_size=150, window=10, min_count=3, n_jobs=2, method='skip-gram')
    :param model_fp:
    :return:
    """
    model = word2vec.Word2Vec.load(model_fp)
    return model


if __name__ == '__main__':
    # cid2smiles.txt = '../big-data/all_cid2smiles/all_data_set_CID2Canonical_SMILES.txt'
    # cid_list = '../big-data/all_cid2smiles/step5_x_training_set.csv'
    root_dir = '../big-data/moses_dataset/model_mol2vec/'
    result_file_path1 = os.path.join('../big-data/moses_dataset/nn/cid2smiles_all_in_train_test.csv')
    result_file_path2 = os.path.join(root_dir, 'cid2smiles_training_set_coupus.tmp')
    result_file_path3 = os.path.join(root_dir, 'cid2smiles_training_set_coupus.txt')
    mol2vec_fp = os.path.join(root_dir, 'model_mol2vec_mol2vec.csv')
    model_fp = os.path.join(root_dir, 'mol2vec_model.pkl')
    # cid2smiles_test = '../big-data/cid2smiles_test.txt'
    # result_file_path4 = '../big-data/vectors/mol2vec_model_mol2vec.csv'
    # get_cid2smiles(cid2smiles.txt, cid_list, result_file=reuslt_file_path1)

    # step1 generate corpus (sentence)
    generate_corpus_from_smiles(in_file=result_file_path1, out_file=result_file_path2, r=1, n_jobs=4)

    # step2 Handling of uncommon "words"
    insert_unk(corpus=result_file_path2, out_corpus=result_file_path3)

    # step3 train molecule vector
    train_word2vec_model(infile_name=result_file_path3, outfile_name=model_fp,
                         vector_size=100, window=10, min_count=3, n_jobs=4, method='cbow')

    # get vector of each molecule by mol2vec model
    # mol with fragment id sentence
    mol_info = pd.read_csv(result_file_path3, header=None)

    # model_fp = os.path.join(include_small_dataset_dir, 'mol2vec_related', 'mol2vec_model.pkl')
    model = load_trained_model(model_fp)
    # print(mol_info.loc[4568802, '0'])
    mol_info['sentence'] = mol_info.apply(lambda x: MolSentence([str(i) for i in x[0].split(' ')]), axis=1)
    # print(mol_info)
    mol_info['mol2vec_related'] = [DfVec(x) for x in sentences2vec(mol_info['sentence'], model)]
    cid2vec = {}
    cid2smiles = pd.read_csv(result_file_path1)
    inx2cid = cid2smiles['0'].to_dict()
    for inx in mol_info.index.to_list():
        cid = inx2cid[inx]
        cid2vec[cid] = list(mol_info.loc[inx, 'mol2vec_related'].vec)
    cid2vec_df = pd.DataFrame.from_dict(cid2vec, orient='index')
    print(cid2vec_df.shape)
    # result_file2 = os.path.join(result_dir, 'step4_selected_mol2vec_model_mol2vec.csv')
    cid2vec_df.to_csv(mol2vec_fp, header=False, float_format='%.3f')